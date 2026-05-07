import json
import time

import requests


class RemoteGPUWorker:
    """Adapter that makes an HTTP worker look like a local GPUworker."""

    def __init__(self, id, url, timeout=180, connect_timeout=5):
        self.id = id
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._last_status = {
            "id": id,
            "is_processing": False,
            "active_requests": 0,
            "capacity": 1,
            "utilization": 0.0,
            "total_processed": 0,
            "avg_time": 0.0,
            "consecutive_failures": 0,
        }

    def process(self, request):
        start = time.time()
        response = requests.post(
            f"{self.url}/process",
            json={"id": request.id, "query": request.query},
            timeout=(self.connect_timeout, self.timeout),
        )
        response.raise_for_status()

        payload = response.json()
        payload.setdefault("id", request.id)
        payload.setdefault("worker_id", self.id)
        payload.setdefault("latency", time.time() - start)
        payload.setdefault("gpu_utilization", 0.0)
        return payload

    def get_status(self):
        try:
            response = requests.get(f"{self.url}/status", timeout=5)
            response.raise_for_status()
            payload = response.json()
            payload.setdefault("id", self.id)
            self._last_status = payload
            return payload
        except requests.RequestException:
            return self._last_status


def load_remote_workers(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        worker_configs = json.load(config_file)

    if not worker_configs:
        raise ValueError(f"No workers configured in {config_path}")

    return [
        RemoteGPUWorker(
            id=worker["id"],
            url=worker["url"],
            timeout=worker.get("timeout", 180),
            connect_timeout=worker.get("connect_timeout", 5),
        )
        for worker in worker_configs
    ]
