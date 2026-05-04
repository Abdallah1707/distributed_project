import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from common.metrics import get_metrics_collector


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"

@dataclass
class WorkerMetrics:
    """Metrics for tracking worker performance"""
    worker_id: int
    active_requests: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    last_used: float = field(default_factory=time.time)
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    
    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests
    
    @property
    def load_score(self) -> float:
        """Calculate load score (higher = more loaded)"""
        return self.active_requests + (self.avg_latency * 10)

class LoadBalancer:
    def __init__(self, workers, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.workers = workers
        self.strategy = strategy
        self.index = 0
        self.health_check_interval = 5
        self.failure_recovery_seconds = 15

        # Track metrics per worker
        self.worker_metrics: Dict[int, WorkerMetrics] = {
            w.id: WorkerMetrics(worker_id=w.id) for w in workers
        }
        # Metrics collector
        self.metrics = get_metrics_collector()
        # Start background health monitor
        self.health_monitor = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_monitor.start()
    
    def set_strategy(self, strategy: LoadBalancingStrategy):
        """Change load balancing strategy at runtime"""
        self.strategy = strategy
        print(f"[LoadBalancer] Strategy changed to: {strategy.value}")
    
    def get_next_worker(self, exclude_ids: Optional[Set[int]] = None):
        """Get next worker based on current strategy"""
        exclude_ids = exclude_ids or set()
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(exclude_ids)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(exclude_ids)
        elif self.strategy == LoadBalancingStrategy.LOAD_AWARE:
            return self._load_aware(exclude_ids)
        return self._round_robin(exclude_ids)
    
    def _round_robin(self, exclude_ids: Set[int]):
        """Round Robin - distribute requests evenly in order"""
        attempts = 0
        while attempts < len(self.workers):
            worker = self.workers[self.index]
            self.index = (self.index + 1) % len(self.workers)
            attempts += 1
            if worker.id not in exclude_ids and self.worker_metrics[worker.id].is_healthy:
                return worker
        return None
    
    def _least_connections(self, exclude_ids: Set[int]):
        """Least Connections - choose worker with fewest active requests"""
        min_load = float('inf')
        selected_worker = None
        
        for worker in self.workers:
            if worker.id in exclude_ids:
                continue
            metrics = self.worker_metrics[worker.id]
            if metrics.is_healthy and metrics.active_requests < min_load:
                min_load = metrics.active_requests
                selected_worker = worker
        
        return selected_worker
    
    def _load_aware(self, exclude_ids: Set[int]):
        """Load Aware - consider both active requests and latency"""
        min_score = float('inf')
        selected_worker = None
        
        for worker in self.workers:
            if worker.id in exclude_ids:
                continue
            metrics = self.worker_metrics[worker.id]
            if metrics.is_healthy and metrics.load_score < min_score:
                min_score = metrics.load_score
                selected_worker = worker
        
        return selected_worker
    
    def dispatch(self, request, retries: int = 3, exclude_ids: Optional[Set[int]] = None):
        """Dispatch request to selected worker with retry and fallback support"""
        exclude_ids = exclude_ids or set()
        worker = self.get_next_worker(exclude_ids)
        if worker is None:
            raise RuntimeError("No healthy workers available to process request")
        
        self.worker_metrics[worker.id].active_requests += 1
        self.worker_metrics[worker.id].last_used = time.time()
        
        try:
            response = worker.process(request)
            latency = response.get('latency', 0)
            
            # Update worker metrics after processing
            self.worker_metrics[worker.id].active_requests -= 1
            self.worker_metrics[worker.id].total_requests += 1
            self.worker_metrics[worker.id].total_latency += latency
            self.worker_metrics[worker.id].consecutive_failures = 0
            
            # Record to global metrics collector
            self.metrics.record_request(
                worker_id=worker.id,
                latency=latency,
                success=True
            )
            
            return response
        except Exception as e:
            self.worker_metrics[worker.id].active_requests = max(0, self.worker_metrics[worker.id].active_requests - 1)
            self.worker_metrics[worker.id].consecutive_failures += 1
            self.worker_metrics[worker.id].last_failure_time = time.time()
            self.mark_worker_unhealthy(worker.id)
            
            self.metrics.record_request(
                worker_id=worker.id,
                latency=0,
                success=False,
                error=str(e)
            )

            if retries > 0:
                print(f"[LoadBalancer] Worker {worker.id} failed; retrying request {request.id} on another worker ({retries} retries left)")
                exclude_ids.add(worker.id)
                return self.dispatch(request, retries=retries - 1, exclude_ids=exclude_ids)
            
            raise RuntimeError(f"Request {request.id} failed after retries: {e}")
    
    def get_worker_stats(self) -> Dict:
        """Get current statistics for all workers"""
        stats = {}
        for worker_id, metrics in self.worker_metrics.items():
            stats[worker_id] = {
                'active_requests': metrics.active_requests,
                'total_requests': metrics.total_requests,
                'avg_latency': metrics.avg_latency,
                'load_score': metrics.load_score,
                'is_healthy': metrics.is_healthy,
                'consecutive_failures': metrics.consecutive_failures
            }
        return stats
    
    def mark_worker_unhealthy(self, worker_id: int):
        """Mark a worker as unhealthy (for fault tolerance)"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].is_healthy = False
            self.worker_metrics[worker_id].last_failure_time = time.time()
            print(f"[LoadBalancer] Worker {worker_id} marked as unhealthy")
    
    def mark_worker_healthy(self, worker_id: int):
        """Mark a worker as healthy again"""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].is_healthy = True
            self.worker_metrics[worker_id].consecutive_failures = 0
            print(f"[LoadBalancer] Worker {worker_id} marked as healthy")
    
    def _health_check_loop(self):
        while True:
            time.sleep(self.health_check_interval)
            now = time.time()
            for worker_id, metrics in self.worker_metrics.items():
                if not metrics.is_healthy:
                    if metrics.last_failure_time and now - metrics.last_failure_time >= self.failure_recovery_seconds:
                        self.mark_worker_healthy(worker_id)
                        print(f"[LoadBalancer] Worker {worker_id} recovered after cooldown")

# Backwards compatibility alias
class loadbalancer(LoadBalancer):
    def __init__(self, workers, strategy: str = "round_robin"):
        # Convert string strategy to enum
        strategy_map = {
            "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
            "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
            "load_aware": LoadBalancingStrategy.LOAD_AWARE
        }
        super().__init__(workers, strategy_map.get(strategy, LoadBalancingStrategy.ROUND_ROBIN))
