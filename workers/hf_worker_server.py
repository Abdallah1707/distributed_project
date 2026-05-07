import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm.inferance import HF_MAX_NEW_TOKENS, HF_MODEL_NAME, device, run_llm
from rag.retriever import retrieve_context


WORKER_ID = int(os.getenv("WORKER_ID", "0"))
WORKER_CAPACITY = int(os.getenv("WORKER_CAPACITY", "1"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "1"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "350"))

app = FastAPI(title=f"Hugging Face LLM Worker {WORKER_ID}")
slots = threading.Semaphore(WORKER_CAPACITY)
lock = threading.RLock()
active_requests = 0
total_processed = 0
total_time = 0.0
consecutive_failures = 0


class ProcessRequest(BaseModel):
    id: int
    query: str


@app.get("/health")
def health():
    return {
        "healthy": True,
        "worker_id": WORKER_ID,
        "model": HF_MODEL_NAME,
        "device": str(device),
        "max_new_tokens": HF_MAX_NEW_TOKENS,
        "rag_top_k": RAG_TOP_K,
    }


@app.get("/status")
def status():
    with lock:
        avg_time = total_time / total_processed if total_processed else 0
        utilization = active_requests / WORKER_CAPACITY if WORKER_CAPACITY else 0
        return {
            "id": WORKER_ID,
            "is_processing": active_requests > 0,
            "active_requests": active_requests,
            "capacity": WORKER_CAPACITY,
            "utilization": utilization,
            "total_processed": total_processed,
            "avg_time": avg_time,
            "consecutive_failures": consecutive_failures,
        }


@app.post("/process")
def process(request: ProcessRequest):
    global active_requests, total_processed, total_time, consecutive_failures

    acquired = slots.acquire(timeout=5)
    if not acquired:
        raise HTTPException(status_code=429, detail="Worker is at capacity")

    start = time.time()
    with lock:
        active_requests += 1

    try:
        context = retrieve_context(request.query, top_k=RAG_TOP_K)[:MAX_CONTEXT_CHARS]
        result = run_llm(request.query, context)
        latency = time.time() - start

        with lock:
            total_processed += 1
            total_time += latency
            consecutive_failures = 0
            utilization = active_requests / WORKER_CAPACITY if WORKER_CAPACITY else 0

        return {
            "id": request.id,
            "result": result,
            "latency": latency,
            "worker_id": WORKER_ID,
            "gpu_utilization": utilization,
        }
    except Exception as exc:
        with lock:
            consecutive_failures += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        with lock:
            active_requests = max(0, active_requests - 1)
        slots.release()
