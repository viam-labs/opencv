package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"go.viam.com/rdk/logging"
	"go.viam.com/test"
)

func newTestEndpoint(t *testing.T) (*Endpoint, *Endpoint, func()) {
	t.Helper()
	// l2r: local writes → remote reads. r2l: reverse. Easy to invert by accident.
	l2rReader, l2rWriter := io.Pipe()
	r2lReader, r2lWriter := io.Pipe()

	local := NewEndpoint(logging.NewTestLogger(t), r2lReader, l2rWriter, 0)
	remote := NewEndpoint(logging.NewTestLogger(t), l2rReader, r2lWriter, 0)

	shutdown := func() {
		_ = l2rWriter.Close()
		_ = r2lWriter.Close()
	}
	return local, remote, shutdown
}

func serveFrom(t *testing.T, handle func(msg message) message) (*Endpoint, func()) {
	t.Helper()
	serverReadFromClient, clientWriteToServer := io.Pipe()
	clientReadFromServer, serverWriteToClient := io.Pipe()

	e := NewEndpoint(logging.NewTestLogger(t), clientReadFromServer, clientWriteToServer, 0)

	stop := make(chan struct{})
	go func() {
		r := bufio.NewReader(serverReadFromClient)
		for {
			body, err := readFrame(r)
			if err != nil {
				return
			}
			var req message
			if err := json.Unmarshal(body, &req); err != nil {
				return
			}
			// Handler runs in its own goroutine so a blocking handler (e.g.
			// simulating "server never responds") doesn't stall shutdown.
			go func(req message) {
				respCh := make(chan message, 1)
				go func() { respCh <- handle(req) }()
				select {
				case resp := <-respCh:
					resp.JSONRPC = jsonRPCVersion
					resp.ID = req.ID
					_ = writeFrameTo(serverWriteToClient, resp)
				case <-stop:
				}
			}(req)
		}
	}()

	shutdown := func() {
		close(stop)
		_ = clientWriteToServer.Close()
		_ = serverWriteToClient.Close()
	}
	return e, shutdown
}

func writeFrameTo(w io.Writer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "Content-Length: "+itoa(len(data))+"\r\n\r\n"); err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

func TestCallRoundTrip(t *testing.T) {
	client, shutdown := serveFrom(t, func(req message) message {
		test.That(t, req.Method, test.ShouldEqual, "echo")
		return message{Result: req.Params}
	})
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	raw, err := client.Call(ctx, "echo", map[string]string{"msg": "hi"})
	test.That(t, err, test.ShouldBeNil)

	var got map[string]string
	test.That(t, json.Unmarshal(raw, &got), test.ShouldBeNil)
	test.That(t, got["msg"], test.ShouldEqual, "hi")
}

func TestCallConcurrentCorrelation(t *testing.T) {
	// Delays are intentionally out of order to guarantee out-of-order responses.
	client, shutdown := serveFrom(t, func(req message) message {
		var p struct {
			DelayMs int `json:"delay_ms"`
		}
		_ = json.Unmarshal(req.Params, &p)
		time.Sleep(time.Duration(p.DelayMs) * time.Millisecond)
		return message{Result: json.RawMessage(`"` + req.Method + `"`)}
	})
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for i, delay := range []int{80, 40, 10, 60, 20} {
		wg.Add(1)
		go func(i, delay int) {
			defer wg.Done()
			raw, err := client.Call(ctx, "call"+itoa(i), map[string]int{"delay_ms": delay})
			test.That(t, err, test.ShouldBeNil)
			var got string
			test.That(t, json.Unmarshal(raw, &got), test.ShouldBeNil)
			test.That(t, got, test.ShouldEqual, "call"+itoa(i))
		}(i, delay)
	}
	wg.Wait()
}

func TestCallReturnsRPCError(t *testing.T) {
	client, shutdown := serveFrom(t, func(req message) message {
		return message{Error: &RPCError{
			Code:    -32000,
			Message: "arm halted",
			Data:    json.RawMessage(`{"kind":"arm_halted"}`),
		}}
	})
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := client.Call(ctx, "plan", nil)
	test.That(t, err, test.ShouldNotBeNil)

	var rpcErr *RPCError
	test.That(t, errors.As(err, &rpcErr), test.ShouldBeTrue)
	test.That(t, rpcErr.Kind(), test.ShouldEqual, "arm_halted")
}

func TestCallContextCancel(t *testing.T) {
	client, shutdown := serveFrom(t, func(req message) message { select {} })
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Call(ctx, "wait", nil)
	test.That(t, errors.Is(err, context.DeadlineExceeded), test.ShouldBeTrue)
}

func TestCallFailsWhenReaderClosed(t *testing.T) {
	sr, cw := io.Pipe()
	cr, sw := io.Pipe()
	client := NewEndpoint(logging.NewTestLogger(t), cr, cw, 0)

	errCh := make(chan error, 1)
	go func() {
		_, err := client.Call(context.Background(), "wait", nil)
		errCh <- err
	}()

	go func() {
		buf := make([]byte, 1024)
		for {
			if _, err := sr.Read(buf); err != nil {
				return
			}
		}
	}()

	_ = sw.Close()

	select {
	case err := <-errCh:
		test.That(t, err, test.ShouldNotBeNil)
	case <-time.After(2 * time.Second):
		t.Fatal("call did not fail after reader closed")
	}
	<-client.Done()
	_ = cw.Close()
}

func TestReadFrameRejectsMissingContentLength(t *testing.T) {
	_, err := readFrame(bufio.NewReader(strings.NewReader("\r\n\r\nbody")))
	test.That(t, err, test.ShouldNotBeNil)
}

func TestReadFrameRejectsOversizedContentLength(t *testing.T) {
	header := "Content-Length: 99999999999\r\n\r\n"
	_, err := readFrame(bufio.NewReader(strings.NewReader(header)))
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "exceeds max")
}

func TestServerDispatchesHandler(t *testing.T) {
	local, remote, shutdown := newTestEndpoint(t)
	defer shutdown()

	remote.Register("add", func(ctx context.Context, params json.RawMessage) (json.RawMessage, *RPCError) {
		var p struct{ A, B int }
		if err := json.Unmarshal(params, &p); err != nil {
			return nil, &RPCError{Code: -32602, Message: err.Error()}
		}
		return json.RawMessage(itoa(p.A + p.B)), nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	raw, err := local.Call(ctx, "add", map[string]int{"A": 2, "B": 3})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, string(raw), test.ShouldEqual, "5")
}

func TestServerUnknownMethodReturnsError(t *testing.T) {
	local, _, shutdown := newTestEndpoint(t)
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := local.Call(ctx, "nope", nil)
	test.That(t, err, test.ShouldNotBeNil)

	var rpcErr *RPCError
	test.That(t, errors.As(err, &rpcErr), test.ShouldBeTrue)
	test.That(t, rpcErr.Code, test.ShouldEqual, -32601)
}

func TestBidirectionalTraffic(t *testing.T) {
	local, remote, shutdown := newTestEndpoint(t)
	defer shutdown()

	local.Register("ping", func(ctx context.Context, _ json.RawMessage) (json.RawMessage, *RPCError) {
		return json.RawMessage(`"pong-from-local"`), nil
	})
	remote.Register("ping", func(ctx context.Context, _ json.RawMessage) (json.RawMessage, *RPCError) {
		return json.RawMessage(`"pong-from-remote"`), nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	toRemote, err := local.Call(ctx, "ping", nil)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, string(toRemote), test.ShouldEqual, `"pong-from-remote"`)

	toLocal, err := remote.Call(ctx, "ping", nil)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, string(toLocal), test.ShouldEqual, `"pong-from-local"`)
}

func TestBackpressureBoundsInflight(t *testing.T) {
	const maxOut = 3

	sr, cw := io.Pipe()
	cr, sw := io.Pipe()
	client := NewEndpoint(logging.NewTestLogger(t), cr, cw, maxOut)

	go func() {
		buf := make([]byte, 4096)
		for {
			if _, err := sr.Read(buf); err != nil {
				return
			}
		}
	}()

	for i := 0; i < maxOut; i++ {
		go func() { _, _ = client.Call(context.Background(), "block", nil) }()
	}
	time.Sleep(50 * time.Millisecond)

	// One more Call should hit its own deadline waiting for a semaphore slot.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := client.Call(ctx, "extra", nil)
	test.That(t, errors.Is(err, context.DeadlineExceeded), test.ShouldBeTrue)

	_ = sw.Close()
	_ = cw.Close()
}
