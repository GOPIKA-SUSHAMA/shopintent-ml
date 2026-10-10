
import json
import time
import numpy as np
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/predict"

payload = {
    "Administrative": 1,
    "Administrative_Duration": 20,
    "Informational": 0,
    "Informational_Duration": 0,
    "ProductRelated": 12,
    "ProductRelated_Duration": 300,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "SpecialDay": 0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}

body = json.dumps(payload).encode("utf-8")


def send_request(_):
    request = Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=10) as response:
            result = json.load(response)

        assert "purchase_probability" in result

        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, None

    except Exception as exc:
        return None, str(exc)


def benchmark(total_requests=50, workers=1):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            send_request, range(total_requests)
        ))

    duration = time.perf_counter() - start

    latencies = np.array([
        latency for latency, error in results
        if error is None
    ])

    errors = [
        error for latency, error in results
        if error is not None
    ]

    print(f"\nConcurrent workers: {workers}")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Failed requests: {len(errors)}")

    if len(latencies):
        print(f"Average latency: {np.mean(latencies):.2f} ms")
        print(f"p50 latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"p95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"p99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"Throughput: {len(latencies) / duration:.2f} req/sec")

    if errors:
        print("First error:", errors[0])


# Warm up the API
for _ in range(5):
    send_request(None)

# Compare sequential and concurrent performance
benchmark(total_requests=50, workers=1)
benchmark(total_requests=50, workers=5)
