from workers.gpu_worker import GPUworker
from lb.load_balancer import LoadBalancer, LoadBalancingStrategy
from master.scheduler import Scheduler
from common.metrics import reset_metrics, get_metrics_collector
from common.models import Request

reset_metrics()
workers = [GPUworker(i, failure_rate=0.3) for i in range(4)]
lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
scheduler = Scheduler(lb)
print('Starting fault tolerance test...')
for i in range(6):
    req = Request(id=i, query=f'Fault tolerance test {i}')
    resp = scheduler.handle_request(req)
    print('RESULT', resp)
print('SUMMARY', get_metrics_collector().get_system_summary())
