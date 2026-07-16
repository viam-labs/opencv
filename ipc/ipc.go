package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"go.viam.com/rdk/logging"
)

const jsonRPCVersion = "2.0"

type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      uint64          `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      uint64          `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("rpc error %d: %s", e.Code, e.Message)
}

// Kind returns the structured error kind (e.g. "plan_infeasible", "arm_halted")
// when present in the Data payload. Returns "" if no kind field is set.
func (e *RPCError) Kind() string {
	if len(e.Data) == 0 {
		return ""
	}
	var payload struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(e.Data, &payload); err != nil {
		return ""
	}
	return payload.Kind
}

type Client struct {
	logger logging.Logger
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	writeM sync.Mutex

	nextID  uint64
	pending map[uint64]chan response
	pendM   sync.Mutex

	done chan struct{}
}

// Spawn starts the Python auxiliary process and begins reading responses.
// socketPath is the viam-server module socket the Python side connects back to
// for dependency resolution.
func Spawn(ctx context.Context, logger logging.Logger, socketPath string, name string, args ...string) (*Client, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = append(os.Environ(), "VIAM_MODULE_SOCKET="+socketPath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start python: %w", err)
	}

	c := &Client{
		logger:  logger,
		cmd:     cmd,
		stdin:   stdin,
		pending: make(map[uint64]chan response),
		done:    make(chan struct{}),
	}

	go c.forwardStderr(stderr)
	go c.readLoop(stdout)

	return c, nil
}

// Call sends a JSON-RPC request and blocks until the response arrives, ctx is
// cancelled, or the subprocess exits.
func (c *Client) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.reserveID()
	ch := make(chan response, 1)

	c.pendM.Lock()
	c.pending[id] = ch
	c.pendM.Unlock()
	defer c.releasePending(id)

	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal params: %w", err)
	}
	req := request{JSONRPC: jsonRPCVersion, ID: id, Method: method, Params: body}
	if err := c.writeFrame(req); err != nil {
		return nil, err
	}

	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.done:
		return nil, errors.New("python subprocess exited")
	}
}

func (c *Client) Close() error {
	_ = c.stdin.Close()
	err := c.cmd.Wait()
	close(c.done)
	return err
}

func (c *Client) reserveID() uint64 {
	c.pendM.Lock()
	defer c.pendM.Unlock()
	c.nextID++
	return c.nextID
}

func (c *Client) releasePending(id uint64) {
	c.pendM.Lock()
	delete(c.pending, id)
	c.pendM.Unlock()
}

func (c *Client) writeFrame(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal frame: %w", err)
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))

	c.writeM.Lock()
	defer c.writeM.Unlock()
	if _, err := c.stdin.Write([]byte(header)); err != nil {
		return err
	}
	if _, err := c.stdin.Write(data); err != nil {
		return err
	}
	return nil
}

func (c *Client) readLoop(stdout io.Reader) {
	defer close(c.done)
	r := bufio.NewReader(stdout)
	for {
		body, err := readFrame(r)
		if err != nil {
			if err != io.EOF {
				c.logger.Errorf("ipc read loop terminated: %v", err)
			}
			return
		}
		var resp response
		if err := json.Unmarshal(body, &resp); err != nil {
			c.logger.Errorf("ipc: malformed response frame: %v", err)
			continue
		}
		c.pendM.Lock()
		ch, ok := c.pending[resp.ID]
		c.pendM.Unlock()
		if !ok {
			c.logger.Warnf("ipc: response for unknown id %d", resp.ID)
			continue
		}
		ch <- resp
	}
}

func (c *Client) forwardStderr(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 8192), 1024*1024)
	for scanner.Scan() {
		c.logger.Infof("[python] %s", scanner.Text())
	}
}

func readFrame(r *bufio.Reader) ([]byte, error) {
	length := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "Content-Length:")))
			if err != nil {
				return nil, fmt.Errorf("bad Content-Length: %w", err)
			}
			length = n
		}
	}
	if length < 0 {
		return nil, errors.New("frame missing Content-Length")
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}
