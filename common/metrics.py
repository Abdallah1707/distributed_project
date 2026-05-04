import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics


@dataclass
class SystemMetrics:
    """Container for system-wide metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def throughput(self) -> float:
        """Requests per second"""
        if self.elapsed_time == 0:
            return 0
        return self.total_requests / self.elapsed_time
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return (self.successful_requests / self.total_requests) * 100


class MetricsCollector:
    """Collects and tracks performance metrics for the distributed system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.lock = threading.RLock()
        
        # System-level metrics
        self.system_metrics = SystemMetrics()
        
        # Per-worker metrics
        self.worker_latencies: Dict[int, List[float]] = defaultdict(list)
        self.worker_throughput: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Recent latencies for percentile calculation (circular buffer)
        self.recent_latencies = deque(maxlen=max_history)
        
        # Request timestamps for throughput calculation
        self.request_times = deque(maxlen=max_history)
        
        # Error tracking
        self.error_types: Dict[str, int] = defaultdict(int)
        
    def record_request(self, worker_id: int, latency: float, 
                       success: bool = True, error: str = None):
        """Record a completed request"""
        with self.lock:
            # System metrics
            self.system_metrics.total_requests += 1
            if success:
                self.system_metrics.successful_requests += 1
            else:
                self.system_metrics.failed_requests += 1
                if error:
                    self.error_types[error] += 1
            
            self.system_metrics.total_latency += latency
            
            # Recent latencies
            self.recent_latencies.append(latency)
            self.request_times.append(time.time())
            
            # Worker-specific metrics
            self.worker_latencies[worker_id].append(latency)
            
    def get_percentile(self, percentile: float) -> float:
        """Calculate latency percentile (e.g., 95 for P95)"""
        with self.lock:
            if not self.recent_latencies:
                return 0.0
            sorted_latencies = sorted(self.recent_latencies)
            index = int(len(sorted_latencies) * (percentile / 100))
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def get_worker_metrics(self, worker_id: int) -> Dict:
        """Get metrics for a specific worker"""
        with self.lock:
            latencies = self.worker_latencies.get(worker_id, [])
            if not latencies:
                return {
                    'worker_id': worker_id,
                    'total_requests': 0,
                    'avg_latency': 0,
                    'min_latency': 0,
                    'max_latency': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0
                }
            
            return {
                'worker_id': worker_id,
                'total_requests': len(latencies),
                'avg_latency': statistics.mean(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'p50': self._calculate_percentile(latencies, 50),
                'p95': self._calculate_percentile(latencies, 95),
                'p99': self._calculate_percentile(latencies, 99)
            }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from a list of values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_system_summary(self) -> Dict:
        """Get complete system metrics summary"""
        with self.lock:
            return {
                'total_requests': self.system_metrics.total_requests,
                'successful_requests': self.system_metrics.successful_requests,
                'failed_requests': self.system_metrics.failed_requests,
                'success_rate': self.system_metrics.success_rate,
                'avg_latency': (
                    self.system_metrics.total_latency / 
                    self.system_metrics.total_requests
                    if self.system_metrics.total_requests > 0 else 0
                ),
                'throughput_rps': self.system_metrics.throughput,
                'p50_latency': self.get_percentile(50),
                'p95_latency': self.get_percentile(95),
                'p99_latency': self.get_percentile(99),
                'elapsed_time': self.system_metrics.elapsed_time,
                'errors': dict(self.error_types)
            }
    
    def get_load_balance_score(self) -> float:
        """Calculate how evenly requests are distributed"""
        with self.lock:
            worker_counts = {
                wid: len(lats) 
                for wid, lats in self.worker_latencies.items()
            }
            
            if not worker_counts or sum(worker_counts.values()) == 0:
                return 0.0
            
            counts = list(worker_counts.values())
            avg = sum(counts) / len(counts)
            if avg == 0:
                return 0.0
            
            # Calculate coefficient of variation
            std_dev = statistics.stdev(counts) if len(counts) > 1 else 0
            return std_dev / avg if avg > 0 else 0
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.system_metrics = SystemMetrics()
            self.worker_latencies.clear()
            self.worker_throughput.clear()
            self.recent_latencies.clear()
            self.request_times.clear()
            self.error_types.clear()


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics():
    """Reset the global metrics collector"""
    global _global_collector
    if _global_collector:
        _global_collector.reset()