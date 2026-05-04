import threading
from common.models import Request

def simulate_user(scheduler, user_id):
    request = Request(id=user_id, query=f"Query {user_id}")
    response = scheduler.handle_request(request)
    if response.get("error"):
        print(f"[Client] Request {request.id} failed: {response['error']}")
    else:
        print(f"[Client] Response {response['id']} | Latency: {response['latency']:.3f}s")

def run_load_test(scheduler, num_users=100):
    """Run concurrent load test using threading"""
    threads = []
    
    for i in range(num_users):
        t = threading.Thread(target=simulate_user, args=(scheduler, i))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    
    