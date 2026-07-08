FROM python:3.12.8-slim

WORKDIR /app

COPY requirements.txt .

# Install everything in ONE layer so build-essential (needed only to compile
# wheels) is purged in the same layer and never persists in the image. This
# keeps the final image small enough to build/unpack on a constrained disk.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY . .

# PORT env (deployment sets it; default 8000 in src/config.py) decides the
# listen port — this EXPOSE is documentation and must match the deployment.
EXPOSE 8001

ENTRYPOINT ["python", "src/app.py"]