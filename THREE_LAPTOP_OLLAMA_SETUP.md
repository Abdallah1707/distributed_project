# Three-Laptop Hugging Face Deployment Plan

This repo is configured so Laptop 1 runs the client, scheduler, load balancer, and metrics, while worker laptops run RAG and Hugging Face LLM inference locally.

```text
Laptop 1: Client + Master/Scheduler + Load Balancer + Metrics
Laptop 2: Worker server + RAG + Hugging Face model
Laptop 3: Worker server + RAG + Hugging Face model
```

No API key is required. Public Hugging Face models download once, cache locally, and then run on the worker laptop using PyTorch.

## Laptop 1

Laptop 1 uses:

```text
main.py
workers/remote_gpu_worker.py
config/workers.json
```

Edit `config/workers.json` with the worker laptop IPs:

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

Install Laptop 1 dependency:

```powershell
.\.venv\Scripts\activate
pip install -r requirements-master.txt
```

Run Laptop 1 after workers are running:

```powershell
python main.py --users 100 --concurrency 10 --strategy least_connections --workers-config config/workers.json
```

For single-laptop fallback only:

```powershell
python main.py --local-workers --users 10 --concurrency 1
```

## Worker Laptops

Copy this project to Laptop 2 and Laptop 3. Each worker laptop needs:

```text
common/
llm/
rag/
workers/hf_worker_server.py
requirements-worker.txt
```

Create and activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-worker.txt
```

Recommended fast default model:

```text
distilgpt2
```

Better instruction following but heavier:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The first run downloads the model from Hugging Face. After that it runs from local cache.

## Run This Laptop As Worker 0 For Local Testing

Use two PowerShell windows.

Window 1, start the worker:

```powershell
cd D:\ASU\semestre_8\Distributed\Project\Phase1
.\.venv\Scripts\activate
$env:WORKER_ID="0"
$env:WORKER_CAPACITY="1"
$env:HF_MODEL_NAME="distilgpt2"
$env:HF_MAX_NEW_TOKENS="20"
$env:RAG_TOP_K="1"
$env:MAX_CONTEXT_CHARS="250"
python -m uvicorn workers.hf_worker_server:app --host 127.0.0.1 --port 8001
```

Set `config/workers.json` to:

```json
[
  {
    "id": 0,
    "url": "http://127.0.0.1:8001",
    "connect_timeout": 5,
    "timeout": 180
  }
]
```

Window 2, test health:

```powershell
curl http://127.0.0.1:8001/health
```

Then run the client/master against the local HTTP worker:

```powershell
python main.py --users 5 --concurrency 1 --strategy least_connections --workers-config config/workers.json
```

## Run Laptop 2 Worker

On Laptop 2:

```powershell
.\.venv\Scripts\activate
$env:WORKER_ID="0"
$env:WORKER_CAPACITY="1"
$env:HF_MODEL_NAME="distilgpt2"
$env:HF_MAX_NEW_TOKENS="20"
$env:RAG_TOP_K="1"
$env:MAX_CONTEXT_CHARS="250"
python -m uvicorn workers.hf_worker_server:app --host 0.0.0.0 --port 8001
```

## Run Laptop 3 Worker

On Laptop 3:

```powershell
.\.venv\Scripts\activate
$env:WORKER_ID="1"
$env:WORKER_CAPACITY="1"
$env:HF_MODEL_NAME="distilgpt2"
$env:HF_MAX_NEW_TOKENS="20"
$env:RAG_TOP_K="1"
$env:MAX_CONTEXT_CHARS="250"
python -m uvicorn workers.hf_worker_server:app --host 0.0.0.0 --port 8001
```

## Test Connectivity From Laptop 1

```powershell
curl http://192.168.1.20:8001/health
curl http://192.168.1.21:8001/health
```

Test one worker request:

```powershell
curl -Method POST http://192.168.1.20:8001/process -ContentType "application/json" -Body '{"id":1,"query":"What is RAG?"}'
```

## Firewall

On worker laptops, allow inbound TCP traffic on port `8001`. If Windows asks, allow Uvicorn on private networks.

Admin PowerShell fallback:

```powershell
New-NetFirewallRule -DisplayName "HF LLM Worker 8001" -Direction Inbound -LocalPort 8001 -Protocol TCP -Action Allow
```

## Speed Notes

For one laptop:

```text
WORKER_CAPACITY=1
HF_MAX_NEW_TOKENS=20
RAG_TOP_K=1
MAX_CONTEXT_CHARS=250
```

If the worker has NVIDIA CUDA working, PyTorch will use `cuda` automatically. Otherwise it uses CPU. Check `/health` to see the selected device.

For real AI demos, start small:

```text
5 users -> 10 users -> 20 users
```

For 1000+ requests with real local LLMs, you need strong worker GPUs or many worker machines. The architecture supports it, but the hardware decides the speed.
