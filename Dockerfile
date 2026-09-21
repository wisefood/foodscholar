# ---------------------------------------------------------------------------
# Stage 1 — build the virtualenv
# ---------------------------------------------------------------------------
# Two stages, because `build-essential` is ~230MB and is needed only to compile
# wheels. The single-stage version purged it inside the same RUN, which kept it
# out of the final layer — but it was still present on disk for the whole
# install, and its apt metadata still had to be written and deleted. Here it
# never exists in the same filesystem as the thing that ships.
FROM python:3.12.8-slim AS builder

# No .pyc during install (they are rewritten at runtime anyway) and no pip
# cache, both of which are pure build-time disk.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# git is not a build tool here, it is a transport: requirements.txt installs
# the foodscholar knowledge graph library from a git URL because it has no PyPI
# release yet. It stays in this stage, so unlike the single-stage version there
# is nothing to purge — it never exists in the image that ships. Drop it once
# the library is published as a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# A venv rather than the system site-packages, so stage 2 is one COPY of one
# directory instead of trying to work out which of /usr/local belongs to us.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt constraints.txt ./

# CPU torch first, and then CONSTRAINED for everything after it.
#
# `requirements.txt` never mentions torch; sentence-transformers pulls it in.
# Installing the CPU build first satisfies it — as long as it accepts 2.6.0. If a future bump does not, pip resolves torch from PyPI, which
# means the CUDA wheel: ~800MB to download, several GB unpacked once the nvidia
# dependencies come with it. On a machine with 6GB free that is the difference
# between a build and a full disk.
#
# The constraint turns that from a silent 5GB into a resolver error naming the
# conflict, which is the failure you want.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -c constraints.txt -r requirements.txt

# The embedding model, into a location that SURVIVES.
#
# This warm-up used to download to the default cache (`/root/.cache`) and the
# next line of the same RUN was `rm -rf /root/.cache` — so the build spent the
# time and the bandwidth and then deleted the model, and every container
# downloaded it again on first question. Pointing the cache somewhere explicit
# is what makes the warm-up mean anything.
# TWO models, because the service runs two retrievers over two indices:
#   all-MiniLM-L6-v2   384-dim — the `rag` retriever's article/guideline index
#   BAAI/bge-base-en-v1.5  768-dim — the knowledge graph's chunk index, which
#                          the `kggen` retriever queries through the library
#
# The second one is not optional for a deployment that serves `kggen`. Without
# it the library downloads it on the first graph question, and in a cluster
# with no egress to HuggingFace that download fails, the facade falls back to
# its hash embedder, and retrieval returns a confidently-ranked list of
# nonsense. Keep this in step with KG_EMBED_MODEL.
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HOME=/opt/models
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    SentenceTransformer('BAAI/bge-base-en-v1.5')" \
    && find /opt/models -name '*.h5' -delete \
    && find /opt/models -name '*.ot' -delete \
    && find /opt/models -name '*.msgpack' -delete \
    && rm -rf /opt/models/hub/*/blobs/*.incomplete

# ---------------------------------------------------------------------------
# Stage 2 — the image that ships
# ---------------------------------------------------------------------------
FROM python:3.12.8-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HOME=/opt/models

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app

# Code only, and code is now all there is.
#
# This used to be followed by a VOLUME for /app/src/data, because the LinearRAG
# retriever read a 617MB index from disk: untracked in git, so a clean clone
# built an image with an empty index and nothing said so, and mounted at
# runtime because baking it in made every source edit rewrite a 617MB layer.
#
# The `kggen` retriever that replaced it reads the same Elasticsearch and Neo4j
# stores the graph build writes, so there is no index to ship, mount or keep in
# sync — the graph is reachable over the network like every other backend.
COPY . .

# PORT env (deployment sets it; default 8000 in src/config.py) decides the
# listen port — this EXPOSE is documentation and must match the deployment.
EXPOSE 8001

ENTRYPOINT ["python", "src/app.py"]
