# Three-Laptop Hugging Face Setup

The main setup guide is currently in:

```text
THREE_LAPTOP_OLLAMA_SETUP.md
```

Despite the old filename, it has been updated to the Hugging Face worker flow:

```text
Laptop 1: client + master + load balancer
Laptop 2: RAG + Hugging Face worker server
Laptop 3: RAG + Hugging Face worker server
```

Use `workers/hf_worker_server.py` on worker laptops.
