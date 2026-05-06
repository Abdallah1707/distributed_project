# Efficient Load Balancing and GPU Cluster Task Distribution

This project simulates a distributed LLM/RAG serving system that can accept 1000+ user requests, route them through a load balancer, process them on simulated GPU worker nodes, and report performance/fault-tolerance metrics.

## Architecture

- `client/load_generator.py`: simulates concurrent users with bounded concurrency.
- `master/scheduler.py`: controller layer that receives requests and asks the load balancer to dispatch them.
- `lb/load_balancer.py`: implements Round Robin, Least Connections, and Load-aware routing with retry-based task reassignment.
- `workers/gpu_worker.py`: simulates GPU nodes, worker capacity, utilization, and node failures.
- `rag/retriever.py`: retrieves relevant context from a small FAISS-backed vector index.
- `llm/inferance.py`: runs either fast simulated LLM responses or real `distilgpt2` inference.
- `common/metrics.py`: tracks latency, throughput, success rate, load balance, errors, and simulated GPU utilization.

## Setup

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run 1000+ Request Simulation

By default the LLM uses real local `distilgpt2` inference through Hugging Face Transformers. For 1000+ request scalability tests on a normal laptop, switch to simulation mode with `$env:USE_REAL_LLM="0"`.

```powershell
python main.py --users 1000 --concurrency 100 --workers 10 --worker-capacity 4 --strategy least_connections
```

Available strategies:

```powershell
python main.py --strategy round_robin
python main.py --strategy least_connections
python main.py --strategy load_aware
```

## Fault Tolerance Test

Increase `--failure-rate` to simulate GPU worker failures. Failed requests are retried on other healthy workers. Keep it low for large tests, because a high failure rate can intentionally exhaust the cluster.

```powershell
python main.py --users 100 --concurrency 20 --workers 4 --worker-capacity 2 --failure-rate 0.05
```

The dashboard reports successful/failed requests and worker health at the end of the run.

## Real LLM Mode

Real GPT-2 inference is enabled by default and is much slower on CPU. Use it for small tests only unless you have suitable GPU resources.

```powershell
python main.py --users 10 --concurrency 2 --workers 2
```

Turn fast simulation mode on:

```powershell
$env:USE_REAL_LLM="0"
```
