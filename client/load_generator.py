from concurrent.futures import ThreadPoolExecutor, as_completed
from common.models import Request

def simulate_user(scheduler, user_id):
    request = Request(id=user_id, query=f"Query {user_id}")
    response = scheduler.handle_request(request)
    if response.get("error"):
        print(f"[Client] Request {request.id} failed: {response['error']}")
    else:
        print(f"[Client] Response {response['id']} | Latency: {response['latency']:.3f}s")
    return response

def run_load_test(scheduler, num_users=100, max_concurrent=50):
    """Run a bounded-concurrency load test.

    num_users is the total request count. max_concurrent is the number of
    simultaneous users allowed to be in-flight at once.
    """
    responses = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(simulate_user, scheduler, i)
            for i in range(num_users)
        ]

        for future in as_completed(futures):
            responses.append(future.result())

    return responses
