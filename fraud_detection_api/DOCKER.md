# Docker Setup for Fraud Detection API

This document covers Docker setup, building, and testing the fraud detection API.

## Prerequisites

- Docker 20.10+
- Docker Compose 1.29+

## Files

- `Dockerfile` - Multi-stage build optimized for size (<200MB)
- `docker-compose.yml` - Complete service configuration
- `requirements-docker.txt` - Minimal runtime dependencies
- `.dockerignore` - Build context optimization
- `test_docker.sh` - Comprehensive test script

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f fraud-detector-api

# Stop the service
docker-compose down
```

### Using Docker CLI

```bash
# Build image
docker build -t fraud-detector-api:latest .

# Check image size
docker images fraud-detector-api

# Run container
docker run -d \
  --name fraud-detector \
  -p 8003:8003 \
  -v "$(pwd)/models:/models:ro" \
  -v "$(pwd)/logs:/logs" \
  fraud-detector-api:latest

# Check if running
curl http://localhost:8003/health
```

## Testing

### Using test script (recommended)

```bash
chmod +x test_docker.sh
./test_docker.sh
```

This runs:
1. Image size validation
2. Container startup
3. `/health` endpoint test
4. `/predict` endpoint test with sample transaction
5. `/metrics` endpoint test
6. Cleanup

### Manual testing

```bash
# Health check
curl http://localhost:8003/health

# Prediction request
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "Time": 0.0, "V1": -1.36, "V2": -0.07, "V3": 2.54,
        "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
        "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
        "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
        "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11,
        "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13,
        "V28": -0.02, "Amount": 149.62
      }
    ]
  }'

# Metrics
curl http://localhost:8003/metrics
```

## Image Size Optimization

Multi-stage build achieves <200MB through:

1. **Builder stage**: 
   - Installs build tools
   - Creates virtual environment
   - Installs dependencies

2. **Runtime stage**:
   - Uses `python:3.11-slim` base (~150MB)
   - Copies only `.local` packages (~30-40MB)
   - Excludes build tools, dev dependencies
   - Minimal filesystem footprint

### Size breakdown (approx)
- Base image: 130MB
- Dependencies: 40MB
- App code: <5MB
- **Total: ~170-180MB**

## Volume Mounts

### Models (read-only)
- Mount: `./models:/models:ro`
- Contains: `fraud_detector_xgb_v1.pkl`, `lightgbm_best_model.pkl`
- Permission: Read-only to prevent accidental writes

### Logs
- Mount: `./logs:/logs`
- Contains: Application logs from running predictions
- Permission: Read-write for log aggregation

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8003 | API port |
| HOST | 0.0.0.0 | Bind address |
| MODEL_PATH | /models/fraud_detector_xgb_v1.pkl | Primary model |
| FALLBACK_MODEL_PATH | /models/lightgbm_best_model.pkl | Fallback model |
| LOG_LEVEL | info | Logging verbosity |

## Health Checks

Container includes health checks:
- Interval: 30s
- Timeout: 10s
- Retries: 3
- Start period: 5s

Status can be checked with:
```bash
docker inspect --format='{{.State.Health.Status}}' fraud-detector-api
```

## Troubleshooting

### Port already in use
```bash
docker-compose down
# or change port in docker-compose.yml
```

### Model not found
Ensure `models/` directory exists with required model files:
- `fraud_detector_xgb_v1.pkl`
- `lightgbm_best_model.pkl`

### Logs directory permission denied
```bash
mkdir -p logs
chmod 777 logs
```

### Container exits immediately
```bash
docker-compose logs fraud-detector-api
```

## Performance Tuning

### For higher throughput
- Increase ML thread pool workers (modify `main.py`)
- Use replicas with load balancer (Docker Swarm/Kubernetes)

### For lower latency
- Pre-warm model with dummy predictions
- Use XGBoost (faster than LightGBM for this task)

## Production Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detector
  template:
    metadata:
      labels:
        app: fraud-detector
    spec:
      containers:
      - name: api
        image: fraud-detector-api:latest
        ports:
        - containerPort: 8003
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: logs
          mountPath: /logs
      volumes:
      - name: models
        emptyDir: {}
      - name: logs
        emptyDir: {}
```

### Docker Swarm

```bash
docker service create \
  --name fraud-detector \
  --port 8003:8003 \
  --replicas 3 \
  --mount type=bind,source=$(pwd)/models,target=/models,readonly \
  --mount type=bind,source=$(pwd)/logs,target=/logs \
  fraud-detector-api:latest
```

## Notes

- Container runs as root (consider adding non-root user for production)
- Logs are persisted to `./logs` volume
- Models are read-only to prevent accidental corruption
- Graceful shutdown waits for ongoing predictions
