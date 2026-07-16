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

// newClientServer builds a Client wired to a fake JSON-RPC server driven by
// the provided handle func. Returns the client and a shutdown func.
func newClientServer(t *testing.T, handle func(req request) response) (*Client, func()) {
	t.Helper()

	serverReadFromClient, clientWriteToServer := io.Pipe()
	clientReadFromServer, serverWriteToClient := io.Pipe()

	client := NewClient(logging.NewTestLogger(t), clientReadFromServer, clientWriteToServer)

	stop := make(chan struct{})
	go func() {
		r := bufio.NewReader(serverReadFromClient)
		for {
			body, err := readFrame(r)
			if err != nil {
				return
			}
			var req request
			if err := json.Unmarshal(body, &req); err != nil {
				t.Errorf("server: bad frame: %v", err)
				return
			}
			// Dispatch each request in its own goroutine so a blocking handler
			// (e.g. one that simulates "server never responds") doesn't stall
			// the read loop and shutdown.
			go func(req request) {
				respCh := make(chan response, 1)
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
	return client, shutdown
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
	client, shutdown := newClientServer(t, func(req request) response {
		test.That(t, req.Method, test.ShouldEqual, "echo")
		return response{Result: req.Params}
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
	// Server sleeps proportionally to the id to guarantee out-of-order responses.
	client, shutdown := newClientServer(t, func(req request) response {
		var payload struct {
			DelayMs int `json:"delay_ms"`
		}
		_ = json.Unmarshal(req.Params, &payload)
		time.Sleep(time.Duration(payload.DelayMs) * time.Millisecond)
		return response{Result: json.RawMessage(`"` + req.Method + `"`)}
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
	client, shutdown := newClientServer(t, func(req request) response {
		return response{Error: &RPCError{
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
	// Server never responds.
	client, shutdown := newClientServer(t, func(req request) response {
		select {}
	})
	defer shutdown()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Call(ctx, "wait", nil)
	test.That(t, errors.Is(err, context.DeadlineExceeded), test.ShouldBeTrue)
}

func TestCallFailsWhenReaderClosed(t *testing.T) {
	sr, cw := io.Pipe()
	cr, sw := io.Pipe()
	client := NewClient(logging.NewTestLogger(t), cr, cw)

	// Start a Call that would block waiting for a response.
	errCh := make(chan error, 1)
	go func() {
		_, err := client.Call(context.Background(), "wait", nil)
		errCh <- err
	}()

	// Drain what the client wrote so the pipe doesn't block, then close.
	go func() {
		buf := make([]byte, 1024)
		for {
			if _, err := sr.Read(buf); err != nil {
				return
			}
		}
	}()

	// Simulate subprocess exit by closing what the client reads from.
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
