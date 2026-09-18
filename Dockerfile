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

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# A venv rather than the system site-packages, so stage 2 is one COPY of one
# directory instead of trying to work out which of /usr/local belongs to us.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt constraints.txt ./

# CPU torch first, and then CONSTRAINED for everything after it.
#
# `requirements.txt` never mentions torch; sentence-transformers and scispacy
# pull it in. Installing the CPU build first satisfies them — as long as they
# accept 2.6.0. If a future bump does not, pip resolves torch from PyPI, which
# means the CUDA wheel: ~800MB to download, several GB unpacked once the nvidia
# dependencies come with it. On a machine with 6GB free that is the difference
# between a build and a full disk.
#
# The constraint turns that from a silent 5GB into a resolver error naming the
# conflict, which is the failure you want.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -c constraints.txt -r requirements.txt \
    && pip install -c constraints.txt \
       https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# The embedding model, into a location that SURVIVES.
#
# This warm-up used to download to the default cache (`/root/.cache`) and the
# next line of the same RUN was `rm -rf /root/.cache` — so the build spent the
# time and the bandwidth and then deleted the model, and every container
# downloaded it again on first question. Pointing the cache somewhere explicit
# is what makes the warm-up mean anything.
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HOME=/opt/models
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
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

# Code only. The LinearRAG index is NOT copied — see .dockerignore.
#
# `src/data/linearrag` is 617MB and it was the entire build context: every
# build shipped it to the daemon, wrote it into a layer, and re-wrote that
# layer whenever any source file changed. It is also untracked in git, so the
# image was never reproducible from a checkout anyway — a clean clone built an
# image with an empty index and nothing said so.
#
# It is runtime data, so it is mounted at runtime. `linearrag_service` reads
# `src/data/<dataset_name>`, so the mount target is /app/src/data.
COPY . .

# Present so the mount has somewhere to land, and so the `linearrag` retriever
# fails with "no such file" rather than something stranger. The `rag` and
# `no_rag` retrievers do not touch it and keep working without the mount.
VOLUME ["/app/src/data"]

# PORT env (deployment sets it; default 8000 in src/config.py) decides the
# listen port — this EXPOSE is documentation and must match the deployment.
EXPOSE 8001

ENTRYPOINT ["python", "src/app.py"]
