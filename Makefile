# Build and ship the FoodScholar image.
#
# `make` builds, pushes and verifies, in that order, stopping at the first
# failure. It is one target because the two-step version hid a real problem
# for a day: `build` and `push` were separate, a failed build left the old
# image in place, and `push` then re-pushed that old image and reported
# success. From the outside it looked exactly like a rebuild that changed
# nothing — the registry digest never moved, and the pod kept running a
# client version three releases behind.

DOCKER  = docker
IMAGE   = wisefood/foodscholar
TAG     = latest
#: A second, immutable tag. `latest` alone cannot tell you whether a push
#: landed, because its digest is the only thing that changes; a commit tag
#: can be looked up and is what a rollback asks for.
REV     = $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
CONTEXT = k8s-w
NS      = wf-prod

.PHONY: all build push verify deploy rebuild clean

## Build, push, and prove the image contains what the tree says.
all: build push verify

build:
	@echo "==> building $(IMAGE):$(TAG) ($(REV))"
	# `--pull` so a stale local base image cannot quietly pin an old
	# interpreter or an old certificate bundle.
	$(DOCKER) build --pull . -t $(IMAGE):$(TAG) -t $(IMAGE):$(REV)
	@echo "==> built:"
	@$(DOCKER) images $(IMAGE):$(TAG) --format '    {{.ID}}  {{.CreatedAt}}'

push: build
	@echo "==> pushing $(IMAGE):$(TAG) and $(IMAGE):$(REV)"
	$(DOCKER) push $(IMAGE):$(TAG)
	$(DOCKER) push $(IMAGE):$(REV)

## What the tree asked for against what the image actually has. The pin is
## a `>=`, so "it installed something" is not the same as "it installed the
## release you just published".
verify:
	@echo "==> wanted (requirements.txt):"
	@grep '^wisefood' requirements.txt | sed 's/^/    /'
	@echo "==> got (inside the image):"
	@$(DOCKER) run --rm --entrypoint /opt/venv/bin/pip $(IMAGE):$(TAG) \
		show wisefood | sed -n '1,2p' | sed 's/^/    /' 

## Ignore every cached layer. For when a build succeeds and still ships the
## wrong dependency — the dependency layer is keyed on requirements.txt, so
## an edit that does not change that file will not invalidate it.
rebuild:
	$(DOCKER) build --pull --no-cache . -t $(IMAGE):$(TAG) -t $(IMAGE):$(REV)
	$(MAKE) push verify

## Roll the cluster onto what was just pushed, and say what it is running.
deploy: all
	kubectl --context $(CONTEXT) -n $(NS) rollout restart deploy/foodscholar
	kubectl --context $(CONTEXT) -n $(NS) rollout status deploy/foodscholar --timeout=300s
	@echo "==> running in the cluster:"
	@kubectl --context $(CONTEXT) -n $(NS) exec \
		$$(kubectl --context $(CONTEXT) -n $(NS) get pods -o name | grep foodscholar | head -1) \
		-c fs -- /opt/venv/bin/pip show wisefood | sed -n '1,2p' | sed 's/^/    /'

clean:
	-$(DOCKER) rmi $(IMAGE):$(TAG) $(IMAGE):$(REV)
