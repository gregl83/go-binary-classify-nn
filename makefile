.PHONY: all prepare build install test clean

all: test clean

prepare:
	dep ensure -v

build: prepare
	go build -o worker

install: build

test: install
	go test -v -cover $$(go list ./... | grep -v vendor)

clean:
	go clean