
DOCKER=docker
IMGTAG=wisefood/foodscholar:latest

.PHONY: all build push


all: build push

build:
	$(DOCKER) build . -t $(IMGTAG)

push:
	$(DOCKER) push $(IMGTAG)
