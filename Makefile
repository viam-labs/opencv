.PHONY: default build build-go build-python test clean

MODULE_BINARY := bin/opencv-module

default: build

build: build-go build-python

build-go: $(MODULE_BINARY)

$(MODULE_BINARY): go.mod cmd/module/*.go ipc/*.go forwarding/*.go models/*.go
	go build -o $(MODULE_BINARY) ./cmd/module

build-python:
	./build.sh

test:
	go test -v ./...

clean:
	rm -rf bin dist go_utils/go_utils
