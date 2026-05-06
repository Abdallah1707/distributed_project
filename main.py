import argparse

from workers.gpu_worker import GPUworker
from lb.load_balancer import LoadBalancer, LoadBalancingStrategy
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from common.metrics import get_metrics_collector, reset_metrics


def print_dashboard(metrics):
    """Print a formatted metrics dashboard"""
    summary = metrics.get_system_summary()
    
    print("\n" + "="*70)
    print("                    SYSTEM PERFORMANCE DASHBOARD")
    print("="*70)
    
    # Throughput Section
    print("\nTHROUGHPUT METRICS")
    print("-"*40)
    print(f"  Total Requests:     {summary['total_requests']:,}")
    print(f"  Successful:         {summary['successful_requests']:,}")
    print(f"  Failed:             {summary['failed_requests']:,}")
    print(f"  Success Rate:       {summary['success_rate']:.2f}%")
    print(f"  Throughput:         {summary['throughput_rps']:.2f} req/s")
    print(f"  Elapsed Time:       {summary['elapsed_time']:.2f}s")
    
    # Latency Section
    print("\nLATENCY METRICS")
    print("-"*40)
    print(f"  Average:            {summary['avg_latency']*1000:.2f}ms")
    print(f"  P50 (Median):       {summary['p50_latency']*1000:.2f}ms")
    print(f"  P95:                {summary['p95_latency']*1000:.2f}ms")
    print(f"  P99:                {summary['p99_latency']*1000:.2f}ms")

    print("\nGPU UTILIZATION")
    print("-"*40)
    print(f"  Average:            {summary['avg_gpu_utilization']*100:.2f}%")
    
    # Load Balance Section
    lb_score = metrics.get_load_balance_score()
    print("\nLOAD BALANCE")
    print("-"*40)
    print(f"  Balance Score:      {lb_score:.4f} (lower is better)")
    balance_status = "Excellent" if lb_score < 0.1 else "Good" if lb_score < 0.2 else "Fair" if lb_score < 0.3 else "Poor"
    print(f"  Status:             {balance_status}")
    
    # Worker Details
    print("\nWORKER DISTRIBUTION")
    print("-"*40)
    worker_stats = metrics.worker_latencies
    for worker_id, latencies in worker_stats.items():
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"  Worker {worker_id}: {len(latencies):,} requests | Avg: {avg*1000:.2f}ms")
    
    # Errors
    if summary['errors']:
        print("\nERRORS")
        print("-"*40)
        for error_type, count in summary['errors'].items():
            print(f"  {error_type}: {count}")
    
    print("\n" + "="*70)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed LLM/RAG load-balancing simulation")
    parser.add_argument("--users", type=int, default=1000, help="total number of simulated users/requests")
    parser.add_argument("--concurrency", type=int, default=100, help="maximum simultaneous in-flight requests")
    parser.add_argument("--workers", type=int, default=10, help="number of simulated GPU worker nodes")
    parser.add_argument("--worker-capacity", type=int, default=4, help="parallel request slots per worker")
    parser.add_argument("--failure-rate", type=float, default=0.0, help="simulated worker failure probability per request")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in LoadBalancingStrategy],
        default=LoadBalancingStrategy.LEAST_CONNECTIONS.value,
        help="load-balancing strategy"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Reset metrics for fresh start
    reset_metrics()
    
    # Create GPU workers
    workers = [
        GPUworker(i, failure_rate=args.failure_rate, capacity=args.worker_capacity)
        for i in range(args.workers)
    ]
    
    # Load Balancer with strategy selection
    # Options: ROUND_ROBIN, LEAST_CONNECTIONS, LOAD_AWARE
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy(args.strategy))
    
    # Scheduler
    scheduler = Scheduler(lb)
    
    # Run simulation with configurable 1000+ user load
    print("\n" + "="*60)
    print(f"Starting Load Test with {args.users} Users")
    print(f"Max Concurrent Users: {args.concurrency}")
    print(f"GPU Workers: {args.workers} | Capacity per Worker: {args.worker_capacity}")
    print(f"Load Balancing Strategy: {lb.strategy.value}")
    print("="*60 + "\n")
    
    run_load_test(scheduler, num_users=args.users, max_concurrent=args.concurrency)
    
    # Display comprehensive metrics dashboard
    metrics = get_metrics_collector()
    print_dashboard(metrics)
    
    # Display worker statistics
    print("\n" + "="*60)
    print("Final Worker Statistics:")
    print("="*60)
    stats = lb.get_worker_stats()
    for worker_id, worker_stats in stats.items():
        print(f"Worker {worker_id}:")
        print(f"  - Total Requests: {worker_stats['total_requests']}")
        print(f"  - Active Requests: {worker_stats['active_requests']}")
        print(f"  - Avg Latency: {worker_stats['avg_latency']:.3f}s")
        print(f"  - Load Score: {worker_stats['load_score']:.2f}")
        print(f"  - Capacity: {worker_stats['capacity']}")
        print(f"  - Utilization: {worker_stats['utilization']*100:.1f}%")
        print(f"  - Healthy: {worker_stats['is_healthy']}")
        
    print(f"\nLoad Balancing Strategy: {lb.strategy.value}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
