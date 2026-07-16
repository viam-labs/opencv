package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
// from the Data payload, or "" if absent.
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

// Client is a JSON-RPC 2.0 client over a framed reader/writer pair. It does not
// manage the underlying transport (pipes, sockets, subprocesses) — callers own
// those resources. Close the writer to shut the client down cleanly.
type Client struct {
	logger logging.Logger
	writer io.Writer
	writeM sync.Mutex

	nextID  uint64
	idM     sync.Mutex
	pending map[uint64]chan response
	pendM   sync.Mutex

	done   chan struct{}
	doneM  sync.Mutex
	closed bool
	err    error
}

// NewClient wraps a reader/writer pair as a JSON-RPC client and starts reading
// responses in a background goroutine. The reader is consumed until EOF or a
// framing/protocol error, at which point pending calls fail and the client
// enters a terminal state.
func NewClient(logger logging.Logger, reader io.Reader, writer io.Writer) *Client {
	c := &Client{
		logger:  logger,
		writer:  writer,
		pending: make(map[uint64]chan response),
		done:    make(chan struct{}),
	}
	go c.readLoop(reader)
	return c
}

// Call sends a JSON-RPC request and blocks for the response, ctx cancellation,
// or client termination (reader EOF).
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
	if err := c.writeFrame(request{JSONRPC: jsonRPCVersion, ID: id, Method: method, Params: body}); err != nil {
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
		return nil, c.terminalErr()
	}
}

// Done returns a channel that closes when the read loop exits (EOF or protocol error).
func (c *Client) Done() <-chan struct{} {
	return c.done
}

// Err returns the terminal error that ended the read loop, or nil if still running.
func (c *Client) Err() error {
	return c.terminalErr()
}

func (c *Client) reserveID() uint64 {
	c.idM.Lock()
	defer c.idM.Unlock()
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
	if _, err := c.writer.Write([]byte(header)); err != nil {
		return err
	}
	if _, err := c.writer.Write(data); err != nil {
		return err
	}
	return nil
}

func (c *Client) readLoop(reader io.Reader) {
	r := bufio.NewReader(reader)
	for {
		body, err := readFrame(r)
		if err != nil {
			c.finish(err)
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

func (c *Client) finish(readErr error) {
	c.doneM.Lock()
	if c.closed {
		c.doneM.Unlock()
		return
	}
	c.closed = true
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		c.err = readErr
	}
	c.doneM.Unlock()
	close(c.done)
}

func (c *Client) terminalErr() error {
	c.doneM.Lock()
	defer c.doneM.Unlock()
	if c.closed && c.err == nil {
		return errors.New("ipc client closed")
	}
	return c.err
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
