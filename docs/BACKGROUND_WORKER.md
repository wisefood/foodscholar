# Background Enrichment Worker

A simplified background worker that runs as a thread within your FastAPI application, continuously enriching articles from the data catalog.

## Features

- **Runs in Background**: Operates as a daemon thread in your FastAPI app
- **Multi-Replica Safe**: Uses Redis locks to prevent duplicate processing across replicas
- **Graceful Shutdown**: Stops cleanly when the app shuts down
- **Automatic Retry**: Retries failed articles up to 3 times
- **Simple Configuration**: Just set environment variables

## Quick Start

### 1. Enable the Worker

Set the environment variable:

```bash
export ENABLE_BACKGROUND_WORKER=true
```

### 2. Start Your FastAPI App

```bash
cd src
python app.py
```

The worker will start automatically in a background thread!

### 3. Check Worker Status

Visit the health endpoint:

```bash
curl http://localhost:8000/health
```

Or the dedicated worker status endpoint:

```bash
curl http://localhost:8000/api/v1/worker/status
```

## Configuration

Configure via environment variables:

```bash
# Enable/disable the worker
ENABLE_BACKGROUND_WORKER=true

# Number of articles to fetch per batch
WORKER_BATCH_SIZE=50

# Seconds to wait between polling cycles
WORKER_POLL_INTERVAL=10
```

## How It Works

### Architecture

```
┌─────────────────────────────────┐
│      FastAPI Application        │
│                                 │
│  ┌───────────────────────────┐ │
│  │   Background Thread       │ │
│  │   (Enrichment Worker)     │ │
│  │                           │ │
│  │  1. Fetch articles        │ │
│  │  2. Check Redis lock      │ │
│  │  3. Enrich article        │ │
│  │  4. Update catalog        │ │
│  │  5. Mark processed        │ │
│  └───────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
         │              │
         ▼              ▼
    ┌────────┐    ┌──────────┐
    │ Redis  │    │ WiseFood │
    │ Locks  │    │   API    │
    └────────┘    └──────────┘
```

### Processing Flow

1. **Fetch Batch**: Worker fetches up to 50 articles from the catalog
2. **Check Processed**: Skip articles already in Redis `enrichment:processed` set
3. **Acquire Lock**: Try to acquire Redis lock using `SET article_id NX EX 300`
   - If locked by another worker/replica → skip
   - If acquired → proceed with processing
4. **Enrich**: Run EnrichmentAgent to analyze the article
5. **Update Catalog**: Store enriched data via WiseFood API:
   - `ai_tags` → from keywords
   - `ai_category` → from study_type
   - `ai_key_takeaways` → from evaluation verdict
   - `extras` → all other enrichment data
6. **Mark Complete**: Add article ID to `enrichment:processed` set
7. **Release Lock**: Delete lock key

### Multi-Replica Behavior

When running multiple API replicas (e.g., 3 instances), each replica runs its own background worker thread. Redis locks ensure only one worker processes each article:

```
Replica 1 Worker → tries to lock article 123 → SUCCESS → processes
Replica 2 Worker → tries to lock article 123 → LOCKED → skips
Replica 3 Worker → tries to lock article 123 → LOCKED → skips

Replica 1 Worker → tries to lock article 456 → LOCKED → skips
Replica 2 Worker → tries to lock article 456 → SUCCESS → processes
Replica 3 Worker → tries to lock article 456 → LOCKED → skips
```

## Monitoring

### Check Worker Status

```bash
# Health endpoint (includes worker stats if enabled)
curl http://localhost:8000/health

# Worker-specific endpoint
curl http://localhost:8000/api/v1/worker/status
```

Response:

```json
{
  "enabled": true,
  "processed": 150,
  "failed": 5,
  "skipped": 30,
  "started_at": "2026-01-20T10:30:00",
  "running": true,
  "uptime_seconds": 3600
}
```

### View Logs

The worker logs to the same logger as your FastAPI app:

```bash
# If running locally
tail -f logs/app.log

# If running with Docker
docker logs -f foodscholar-api
```

## Redis Data

The worker uses these Redis keys:

- `enrichment:processed` (SET) - IDs of successfully processed articles
- `enrichment:lock:{article_id}` (STRING, 300s TTL) - Processing locks
- `enrichment:retry:{article_id}` (STRING, 24h TTL) - Retry counts

### Clear Processed Set

To reprocess all articles:

```bash
redis-cli
> DEL enrichment:processed
```

### Check Locks

```bash
redis-cli
> KEYS enrichment:lock:*
```

## Deployment

### Docker Compose

```yaml
services:
  foodscholar-api:
    build: .
    environment:
      - ENABLE_BACKGROUND_WORKER=true
      - WORKER_BATCH_SIZE=50
      - WORKER_POLL_INTERVAL=10
      - REDIS_HOST=redis
      - DATA_API_URL=http://data-api:8000
      - KEYCLOAK_CLIENT_ID=${KEYCLOAK_CLIENT_ID}
      - KEYCLOAK_CLIENT_SECRET=${KEYCLOAK_CLIENT_SECRET}
    depends_on:
      - redis
    deploy:
      replicas: 3  # Multiple replicas work seamlessly

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foodscholar-api
spec:
  replicas: 3  # Scale as needed
  template:
    spec:
      containers:
      - name: api
        image: foodscholar:latest
        env:
        - name: ENABLE_BACKGROUND_WORKER
          value: "true"
        - name: WORKER_BATCH_SIZE
          value: "50"
        - name: REDIS_HOST
          value: "redis-service"
```

## Troubleshooting

### Worker Not Starting

Check logs for errors:

```bash
grep "Background enrichment worker" logs/app.log
```

Verify Redis connection:

```bash
redis-cli -h $REDIS_HOST ping
```

### Articles Not Being Enriched

Check worker status:

```bash
curl http://localhost:8000/api/v1/worker/status
```

Check if articles are being fetched:

```bash
# Look for "Fetched N articles" in logs
grep "Fetched" logs/app.log
```

### High Failure Rate

Check retry counts in Redis:

```bash
redis-cli
> KEYS enrichment:retry:*
> GET enrichment:retry:12345
```

View logs for error details:

```bash
grep "Failed to process article" logs/app.log
```

### Duplicate Processing

This should NOT happen due to Redis locks. If it does:

1. Verify Redis version (needs 2.6.12+ for `SET NX`)
2. Check for clock skew between replicas
3. Verify network connectivity to Redis

## Performance Tuning

### Batch Size

- **Small (10-20)**: Lower memory, more API calls
- **Large (100-200)**: Higher memory, fewer API calls
- **Default: 50** (recommended)

### Poll Interval

- **Short (5s)**: More responsive, higher load
- **Long (30s)**: Less responsive, lower load
- **Default: 10s** (recommended)

### Number of Replicas

More replicas = faster processing (up to a point):

- 1 replica: ~360 articles/hour (assuming 10s per article)
- 3 replicas: ~1080 articles/hour
- 5 replicas: ~1800 articles/hour

## Disable the Worker

To disable temporarily without changing code:

```bash
export ENABLE_BACKGROUND_WORKER=false
```

Or remove the environment variable entirely (defaults to false).

## FAQ

**Q: Can I run this without Redis?**
A: No, Redis is required for distributed locking across replicas.

**Q: What happens if the app crashes?**
A: The worker thread stops, but locks will expire after 5 minutes (300s). Articles can be reprocessed on next startup.

**Q: Can I process only specific articles?**
A: Currently, the worker processes all articles from the catalog. To customize, modify the `_run()` method in `background_enrichment.py` to add filters.

**Q: How do I force reprocessing?**
A: Clear the processed set: `redis-cli DEL enrichment:processed`

**Q: Does this work with auto-scaling?**
A: Yes! You can scale replicas up/down dynamically. Each replica's worker coordinates via Redis.
