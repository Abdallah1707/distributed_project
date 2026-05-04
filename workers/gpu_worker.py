import random
import time
from llm.inferance import run_llm
from rag.retriever import retrieve_context


class GPUworker:
    def __init__(self, id, failure_rate: float = 0.05):
        self.id = id
        self.is_processing = False
        self.current_request_id = None
        self.start_time = None
        self.total_processed = 0
        self.total_time = 0.0
        self.failure_rate = failure_rate
        self.last_failure_time = None
        self.consecutive_failures = 0
        
    def process(self, request):
        self.is_processing = True
        self.current_request_id = request.id
        start = time.time()
        
        print(f"[Worker {self.id}] Processing request: {request.id}")
        
        # Simulated hardware failure or inference error
        if random.random() < self.failure_rate:
            self.last_failure_time = time.time()
            self.consecutive_failures += 1
            self.is_processing = False
            self.current_request_id = None
            raise RuntimeError(f"Simulated GPU failure on worker {self.id}")
        
        # RAG Step
        context = retrieve_context(request.query)
        # LLM Step
        result = run_llm(request.query, context)
        
        latency = time.time() - start
        
        # Update stats
        self.is_processing = False
        self.current_request_id = None
        self.total_processed += 1
        self.total_time += latency
        self.consecutive_failures = 0
        
        return {
            "id": request.id,
            "result": result,
            "latency": latency,
            "worker_id": self.id
        }
    
    def get_status(self):
        """Get current worker status"""
        return {
            "id": self.id,
            "is_processing": self.is_processing,
            "current_request": self.current_request_id,
            "total_processed": self.total_processed,
            "avg_time": self.total_time / self.total_processed if self.total_processed > 0 else 0,
            "consecutive_failures": self.consecutive_failures
        }
