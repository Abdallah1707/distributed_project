import random
import threading
import time
from llm.inferance import run_llm
from rag.retriever import retrieve_context


class GPUworker:
    def __init__(self, id, failure_rate: float = 0.0, capacity: int = 1):
        self.id = id
        self.capacity = capacity
        self._slots = threading.Semaphore(capacity)
        self._lock = threading.RLock()
        self.active_requests = 0
        self.current_request_id = None
        self.start_time = None
        self.total_processed = 0
        self.total_time = 0.0
        self.failure_rate = failure_rate
        self.last_failure_time = None
        self.consecutive_failures = 0
        
    def process(self, request):
        self._slots.acquire()
        start = time.time()
        with self._lock:
            self.active_requests += 1
            self.current_request_id = request.id

        try:
            print(f"[Worker {self.id}] Processing request: {request.id}")

            # Simulated hardware failure or inference error
            if random.random() < self.failure_rate:
                with self._lock:
                    self.last_failure_time = time.time()
                    self.consecutive_failures += 1
                raise RuntimeError(f"Simulated GPU failure on worker {self.id}")

            # RAG Step
            context = retrieve_context(request.query)
            # LLM Step
            result = run_llm(request.query, context)

            latency = time.time() - start

            with self._lock:
                self.total_processed += 1
                self.total_time += latency
                self.consecutive_failures = 0
                utilization = self.active_requests / self.capacity if self.capacity else 0

            return {
                "id": request.id,
                "result": result,
                "latency": latency,
                "worker_id": self.id,
                "gpu_utilization": utilization
            }
        finally:
            with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
                self.current_request_id = None
            self._slots.release()
    
    def get_status(self):
        """Get current worker status"""
        with self._lock:
            avg_time = self.total_time / self.total_processed if self.total_processed > 0 else 0
            utilization = self.active_requests / self.capacity if self.capacity else 0
            is_processing = self.active_requests > 0
            current_request = self.current_request_id
            total_processed = self.total_processed
            consecutive_failures = self.consecutive_failures

        return {
            "id": self.id,
            "is_processing": is_processing,
            "current_request": current_request,
            "active_requests": self.active_requests,
            "capacity": self.capacity,
            "utilization": utilization,
            "total_processed": total_processed,
            "avg_time": avg_time,
            "consecutive_failures": consecutive_failures
        }
