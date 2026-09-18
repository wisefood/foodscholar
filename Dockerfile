FROM python:3.12.8-slim

WORKDIR /app

COPY requirements.txt .

# Install everything in ONE layer so build-essential (needed only to compile
# wheels) is purged in the same layer and never persists in the image. This
# keeps the final image small enough to build/unpack on a constrained disk.
# git is here for one reason: the foodscholar knowledge graph library installs
# from a git URL because it has no PyPI release yet. It is purged in this same
# layer alongside build-essential, so neither reaches the final image. Drop it
# again once the library is published as a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && apt-get purge -y build-essential git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY . .

# PORT env (deployment sets it; default 8000 in src/config.py) decides the
# listen port — this EXPOSE is documentation and must match the deployment.
EXPOSE 8001

ENTRYPOINT ["python", "src/app.py"]