class Scheduler:
    #print(f"scheduler working")
    def __init__(self, load_balancer):
        self.lb = load_balancer

    def handle_request(self, request):
        print(f"[Scheduler] Dispatching request: {request.id}")
        try:
            response = self.lb.dispatch(request)
            return response
        except Exception as e:
            return {
                "id": request.id,
                "result": None,
                "latency": 0.0,
                "worker_id": None,
                "error": str(e)
            }