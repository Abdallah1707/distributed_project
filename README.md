# Distributed LLM/RAG Serving System

A scalable distributed system that simulates or runs real LLM and RAG workloads on GPU clusters. The system handles 1000+ concurrent requests, routes them through intelligent load balancers, and provides comprehensive performance metrics and fault tolerance.

## Project Overview

This system demonstrates a production-grade distributed architecture for serving LLM/RAG applications across multiple worker nodes. It supports:

- **Multiple Load Balancing Strategies**: Round Robin, Least Connections, and Load-aware routing
- **Fault Tolerance**: Automatic retry and reassignment of failed requests
- **Real or Simulated Workloads**: Switch seamlessly between real Hugging Face models and fast simulated responses
- **Real or Simulated GPU Workers**: Test with simulated local workers or distribute across remote worker machines
- **Comprehensive Metrics**: Latency, throughput, success rate, load balance score, and GPU utilization
- **Scalable Architecture**: Handle 100s to 1000s of concurrent requests

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT / LOAD GENERATOR                     │
│                    (load_generator.py)                           │
│               Generates 1000+ concurrent requests                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                      MASTER / SCHEDULER                         │
│                    (scheduler.py)                               │
│         Receives requests and coordinates load balancer         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     LOAD BALANCER                               │
│                 (load_balancer.py)                              │
│    Routes requests using RR / LC / Load-aware strategies        │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
   │ WORKER 0 │      │ WORKER 1 │  ... │ WORKER N │
   │          │      │          │      │          │
   │ ┌──────┐ │      │ ┌──────┐ │      │ ┌──────┐ │
   │ │ RAG  │ │      │ │ RAG  │ │      │ │ RAG  │ │
   │ │Retri │ │      │ │Retri │ │      │ │Retri │ │
   │ └──────┘ │      │ └──────┘ │      │ └──────┘ │
   │ ┌──────┐ │      │ ┌──────┐ │      │ ┌──────┐ │
   │ │ LLM  │ │      │ │ LLM  │ │      │ │ LLM  │ │
   │ │(Real)│ │      │ │(Real)│ │      │ │(Real)│ │
   │ └──────┘ │      │ └──────┘ │      │ └──────┘ │
   └──────────┘      └──────────┘      └──────────┘
   (Remote or Local Workers)
```

### Key Components

- **`client/load_generator.py`**: Simulates concurrent users with configurable concurrency limits
- **`master/scheduler.py`**: Controller that receives requests and interfaces with load balancer
- **`lb/load_balancer.py`**: Implements multiple routing strategies and handles retry logic
- **`workers/gpu_worker.py`**: Simulates GPU nodes with capacity constraints and failure modes
- **`workers/hf_worker_server.py`**: Real worker server for remote deployment (FastAPI + RAG + LLM)
- **`llm/inferance.py`**: Runs either simulated responses or real Hugging Face inference
- **`rag/retriever.py`**: FAISS-backed vector index with PDF document loading for context retrieval
- **`rag/documents/`**: Directory for PDF knowledge base documents
- **`common/metrics.py`**: Tracks comprehensive performance metrics

## RAG System with PDF Documents

The RAG system now supports loading documents from PDF files:

### Adding Your Documents

1. **Place PDF files** in `rag/documents/` directory:
   ```
   rag/
   └── documents/
       ├── README.md
       ├── your_document.pdf
       ├── research_paper.pdf
       └── user_guide.pdf
   ```

2. **Automatic loading**: When the system starts, PDFs are automatically:
   - Scanned and loaded
   - Text extracted from each page
   - Split into overlapping chunks for better context
   - Converted to embeddings for semantic search

3. **Fallback documents**: If no PDFs are found, the system uses sample documents about LLMs and distributed systems

### Creating Sample PDFs

Generate sample PDFs for testing:

```powershell
cd rag/documents
python create_sample_pdfs.py
cd ../..
```

This creates 3 sample PDFs about distributed systems, machine learning, and GPU computing.

### Document Requirements

- **Format**: PDF files only (`.pdf` extension)
- **Content**: Must contain extractable text (not scanned images)
- **Quality**: Best results with clear, readable PDFs
- **Size**: Split very large documents (1000+ pages) for optimal performance

See `rag/documents/README.md` for detailed instructions.

## Quick Start

### Single Machine (Simulated Workers + Simulated LLM)

**Fastest way to test the distributed system without waiting for LLM inference:**

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run with simulated workers and simulated LLM
python main.py --users 1000 --concurrency 50 --workers 4 --worker-capacity 2 --local-workers
```

### Single Machine (Simulated Workers + Real LLM)

**Test real Hugging Face models locally:**

```powershell
python main.py --users 100 --concurrency 10 --workers 4 --worker-capacity 2 --local-workers
```

### Distributed Deployment (3 Laptops)

**Laptop 1: Master/Load Balancer**

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Edit config/workers.json with worker IPs, then run:
python main.py --users 100 --concurrency 10 --strategy least_connections --workers-config config/workers.json
```

**Laptop 2 & 3: Worker Servers**

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Start worker server (runs on port 8001 by default)
python workers/hf_worker_server.py
```

Then update `config/workers.json` on Laptop 1:

```json
[
  {
    "id": 0,
    "url": "http://192.168.1.20:8001",
    "connect_timeout": 5,
    "timeout": 180
  },
  {
    "id": 1,
    "url": "http://192.168.1.21:8001",
    "connect_timeout": 5,
    "timeout": 180
  }
]
```

## Switching Between Real and Simulated Modes

### LLM: Real vs Simulated

The LLM inference mode is controlled by the `USE_REAL_LLM` environment variable.

**Simulated LLM (Fast - 20ms per request):**

```powershell
# Windows PowerShell
$env:USE_REAL_LLM = "0"
python main.py --users 1000 --concurrency 50 --workers 4 --worker-capacity 2 --local-workers
```

**Real Hugging Face LLM (Actual inference):**

```powershell
# Windows PowerShell
$env:USE_REAL_LLM = "1"
# or just run without setting (default is 1)
python main.py --users 100 --concurrency 10 --workers 4 --worker-capacity 2 --local-workers
```

**Configure LLM Model and Parameters:**

```powershell
$env:USE_REAL_LLM = "1"
$env:HF_MODEL_NAME = "distilgpt2"           # Fast model (~500MB)
$env:HF_MAX_NEW_TOKENS = "32"                # Response length
$env:HF_TEMPERATURE = "0.2"                  # Creativity (0-1)
$env:HF_DEVICE = "cpu"                       # "cpu" or "cuda"

python main.py --users 50 --concurrency 5 --workers 4 --worker-capacity 2 --local-workers
```

**Recommended Models:**

- **Fast**: `distilgpt2` (~500MB, ~10ms per token)
- **Better**: `Qwen/Qwen2.5-0.5B-Instruct` (~1.5GB, ~15ms per token)
- **Larger**: `Qwen/Qwen2.5-1.5B-Instruct` (~5GB, ~25ms per token)

### GPU: Real vs Simulated

Worker deployment mode is controlled by flags in `main.py`.

**Simulated GPU Workers (Local):**

Uses the `--local-workers` flag. Simulates GPU nodes with configurable capacity and failure rates:

```powershell
python main.py \
  --users 1000 \
  --concurrency 50 \
  --workers 4 \
  --worker-capacity 2 \
  --local-workers
```

**Real GPU Workers (Remote Servers):**

Uses actual worker servers via `--workers-config`:

```powershell
python main.py \
  --users 100 \
  --concurrency 10 \
  --strategy least_connections \
  --workers-config config/workers.json
```

## Command Line Options

```
Options:
  --users NUM                  Total requests to generate (default: 100)
  --concurrency NUM            Max concurrent requests (default: 10)
  --strategy STR               Load balancing: round_robin, least_connections, load_aware (default: round_robin)
  --workers NUM                Number of simulated workers (default: 4)
  --worker-capacity NUM        Requests per worker (default: 2)
  --failure-rate FLOAT         Simulated failure rate 0.0-1.0 (default: 0.0)
  --local-workers              Use simulated workers instead of remote config (default: False)
  --workers-config PATH        Path to workers.json for remote deployment (default: config/workers.json)
```

## Fault Tolerance Testing

Simulate worker failures and test retry logic:

```powershell
# 5% failure rate - some requests will fail and be retried
python main.py --users 100 --concurrency 20 --workers 4 --worker-capacity 2 --failure-rate 0.05 --local-workers

# Higher failure rate for stress testing
python main.py --users 50 --concurrency 10 --workers 4 --worker-capacity 2 --failure-rate 0.2 --local-workers
```

## Load Balancing Strategies

**Round Robin**: Distributes requests evenly across all workers in sequence.

```powershell
python main.py --strategy round_robin --users 100 --concurrency 10 --local-workers
```

**Least Connections**: Routes to the worker with fewest active requests.

```powershell
python main.py --strategy least_connections --users 100 --concurrency 10 --local-workers
```

**Load-Aware**: Considers both active connections and average response time.

```powershell
python main.py --strategy load_aware --users 100 --concurrency 10 --local-workers
```

## Performance Metrics

The system tracks and displays:

- **Throughput**: Total requests, successful/failed, success rate, requests/second
- **Latency**: Average, P50 (median), P95, P99 response times
- **GPU Utilization**: Average utilization across workers
- **Load Balance**: Score measuring distribution fairness (lower is better)
- **Worker Distribution**: Per-worker request counts and performance

## Project Structure

```
Phase1/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── config/
│   └── workers.json                # Worker configurations
├── client/
│   └── load_generator.py           # Load testing client
├── master/
│   └── scheduler.py                # Request scheduler
├── lb/
│   └── load_balancer.py            # Load balancing logic
├── workers/
│   ├── gpu_worker.py               # Simulated GPU worker
│   ├── hf_worker_server.py         # Real worker server
│   └── remote_gpu_worker.py        # Remote worker interface
├── llm/
│   └── inferance.py                # LLM inference (real/simulated)
├── rag/
│   └── retriever.py                # Vector index + retrieval
└── common/
    ├── metrics.py                  # Performance metrics
    └── models.py                   # Request/Response models
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_REAL_LLM` | `1` | Use real Hugging Face models (0 = simulated) |
| `HF_MODEL_NAME` | `distilgpt2` | Hugging Face model to use |
| `HF_MAX_NEW_TOKENS` | `32` | Max tokens to generate |
| `HF_TEMPERATURE` | `0.2` | Sampling temperature (0=deterministic) |
| `HF_DEVICE` | Auto-detect | `cpu` or `cuda` |

## Example Workflows

### 1. Quick Distributed System Test (Simulated)

```powershell
# Fast test of load balancing without GPU/LLM overhead
$env:USE_REAL_LLM = "0"
python main.py --users 1000 --concurrency 50 --workers 4 --worker-capacity 2 --local-workers
```

### 2. Real LLM Single-Machine Test

```powershell
# Test with actual Hugging Face inference locally
$env:HF_MODEL_NAME = "distilgpt2"
python main.py --users 50 --concurrency 5 --workers 4 --worker-capacity 2 --local-workers
```

### 3. Distributed Cluster Simulation

```powershell
# 100 requests across 3 worker servers with smart routing
python main.py --users 100 --concurrency 10 --strategy load_aware --workers-config config/workers.json
```

### 4. Fault Tolerance Benchmark

```powershell
# Test system resilience with 10% failure rate
python main.py --users 200 --concurrency 20 --failure-rate 0.1 --workers 4 --worker-capacity 2 --local-workers
```

## Dependencies

- **torch**: PyTorch for LLM inference
- **transformers**: Hugging Face model loading
- **sentence-transformers**: Embeddings for RAG
- **faiss-cpu**: Vector index for retrieval
- **fastapi**: Worker server API
- **uvicorn**: ASGI application server
- **requests**: HTTP client for remote workers
- **numpy**: Numerical operations

## Notes

- First run of real LLM will download the model (~500MB - 5GB depending on model)
- Models cache locally in `~/.cache/huggingface/hub/`
- Use `USE_REAL_LLM=0` for CI/CD and testing without infrastructure
- Worker timeouts in `config/workers.json` can be adjusted for different network conditions
- Simulated workers are thread-based; real workers are process-based for isolation
