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

const (
	jsonRPCVersion = "2.0"

	// MaxFrameSize caps the size of an incoming JSON-RPC frame. Guards against
	// a misbehaving peer advertising a huge Content-Length and OOMing us.
	MaxFrameSize = 16 * 1024 * 1024

	// DefaultMaxOutstanding is the default cap on in-flight outgoing calls per
	// endpoint. Bounds the pending map so a slow or hung peer can't grow it
	// without limit.
	DefaultMaxOutstanding = 32
)

type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      uint64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCError is the JSON-RPC 2.0 error object. Data carries a structured `kind`
// field callers use to branch (e.g. "plan_infeasible" vs "arm_halted").
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("rpc error %d: %s", e.Code, e.Message)
}

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

// Handler serves an incoming JSON-RPC request. Return a non-nil *RPCError to
// signal a structured failure; returning a Go error is a bug (use RPCError).
type Handler func(ctx context.Context, params json.RawMessage) (json.RawMessage, *RPCError)

// Endpoint is a bidirectional JSON-RPC 2.0 peer over a framed reader/writer
// pair. Outgoing calls go through Call; incoming calls dispatch to Handlers
// registered via Register. Callers own the underlying transport — close it
// to shut the endpoint down.
type Endpoint struct {
	logger logging.Logger
	writer io.Writer
	writeM sync.Mutex

	nextID  uint64
	idM     sync.Mutex
	pending map[uint64]chan message
	pendM   sync.Mutex

	handlers map[string]Handler
	handlerM sync.RWMutex

	outstanding chan struct{}

	done   chan struct{}
	doneM  sync.Mutex
	closed bool
	err    error
}

// NewEndpoint wraps a reader/writer pair as a bidirectional JSON-RPC endpoint.
// maxOutstanding bounds in-flight outgoing calls; pass 0 for
// DefaultMaxOutstanding.
func NewEndpoint(logger logging.Logger, reader io.Reader, writer io.Writer, maxOutstanding int) *Endpoint {
	if maxOutstanding <= 0 {
		maxOutstanding = DefaultMaxOutstanding
	}
	e := &Endpoint{
		logger:      logger,
		writer:      writer,
		pending:     make(map[uint64]chan message),
		handlers:    make(map[string]Handler),
		outstanding: make(chan struct{}, maxOutstanding),
		done:        make(chan struct{}),
	}
	go e.readLoop(reader)
	return e
}

// Register attaches a handler for an incoming method. Overwrites any prior
// handler for that method.
func (e *Endpoint) Register(method string, h Handler) {
	e.handlerM.Lock()
	e.handlers[method] = h
	e.handlerM.Unlock()
}

// Call sends a JSON-RPC request and blocks for the response, ctx cancellation,
// or endpoint termination. Blocks if maxOutstanding calls are already in
// flight; ctx cancellation aborts the wait.
func (e *Endpoint) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	select {
	case e.outstanding <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-e.done:
		return nil, e.terminalErr()
	}
	defer func() { <-e.outstanding }()

	id := e.reserveID()
	ch := make(chan message, 1)

	e.pendM.Lock()
	e.pending[id] = ch
	e.pendM.Unlock()
	defer e.releasePending(id)

	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal params: %w", err)
	}
	if err := e.writeFrame(message{JSONRPC: jsonRPCVersion, ID: id, Method: method, Params: body}); err != nil {
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
	case <-e.done:
		return nil, e.terminalErr()
	}
}

// Done returns a channel that closes when the read loop exits.
func (e *Endpoint) Done() <-chan struct{} { return e.done }

// Err returns the terminal error that ended the read loop, or nil if running.
func (e *Endpoint) Err() error { return e.terminalErr() }

func (e *Endpoint) reserveID() uint64 {
	e.idM.Lock()
	defer e.idM.Unlock()
	e.nextID++
	return e.nextID
}

func (e *Endpoint) releasePending(id uint64) {
	e.pendM.Lock()
	delete(e.pending, id)
	e.pendM.Unlock()
}

func (e *Endpoint) writeFrame(m message) error {
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal frame: %w", err)
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))

	e.writeM.Lock()
	defer e.writeM.Unlock()
	if _, err := e.writer.Write([]byte(header)); err != nil {
		return err
	}
	if _, err := e.writer.Write(data); err != nil {
		return err
	}
	return nil
}

func (e *Endpoint) readLoop(reader io.Reader) {
	r := bufio.NewReader(reader)
	for {
		body, err := readFrame(r)
		if err != nil {
			e.finish(err)
			return
		}
		var msg message
		if err := json.Unmarshal(body, &msg); err != nil {
			e.logger.Errorf("ipc: malformed frame: %v", err)
			continue
		}
		if msg.Method != "" {
			go e.dispatch(msg)
			continue
		}
		e.pendM.Lock()
		ch, ok := e.pending[msg.ID]
		e.pendM.Unlock()
		if !ok {
			e.logger.Warnf("ipc: response for unknown id %d", msg.ID)
			continue
		}
		ch <- msg
	}
}

func (e *Endpoint) dispatch(req message) {
	e.handlerM.RLock()
	h, ok := e.handlers[req.Method]
	e.handlerM.RUnlock()

	resp := message{JSONRPC: jsonRPCVersion, ID: req.ID}
	if !ok {
		resp.Error = &RPCError{Code: -32601, Message: fmt.Sprintf("unknown method %q", req.Method)}
	} else {
		result, rpcErr := h(context.Background(), req.Params)
		if rpcErr != nil {
			resp.Error = rpcErr
		} else {
			resp.Result = result
		}
	}
	if err := e.writeFrame(resp); err != nil {
		e.logger.Errorf("ipc: write response for %q: %v", req.Method, err)
	}
}

func (e *Endpoint) finish(readErr error) {
	e.doneM.Lock()
	if e.closed {
		e.doneM.Unlock()
		return
	}
	e.closed = true
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		e.err = readErr
	}
	e.doneM.Unlock()
	close(e.done)
}

func (e *Endpoint) terminalErr() error {
	e.doneM.Lock()
	defer e.doneM.Unlock()
	if e.closed && e.err == nil {
		return errors.New("ipc endpoint closed")
	}
	return e.err
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
	if length > MaxFrameSize {
		return nil, fmt.Errorf("frame size %d exceeds max %d", length, MaxFrameSize)
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}
