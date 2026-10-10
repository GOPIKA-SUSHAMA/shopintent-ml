import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

SAMPLE = {
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
    "Weekend": False,
}
BODY = json.dumps(SAMPLE).encode("utf-8")


def send_request(url, timeout=15):
    request = Request(url, data=BODY, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            data = json.load(response)
        probability = data["purchase_probability"]
        if not 0 <= probability <= 1:
            raise ValueError("Invalid purchase probability")
        return (time.perf_counter() - started) * 1000, None
    except Exception as exc:
        return None, str(exc)


def benchmark(url, workers, requests):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: send_request(url), range(requests)))
    duration = time.perf_counter() - start
    samples = np.array([value for value, error in results if error is None])
    errors = [error for _, error in results if error is not None]
    output = {
        "workers": workers, "requests": requests, "successes": int(len(samples)),
        "failures": len(errors), "elapsed_seconds": round(duration, 4),
        "throughput_requests_per_second": round(len(samples) / duration, 3),
    }
    if len(samples):
        output.update({
            "mean_ms": round(float(np.mean(samples)), 3),
            "p50_ms": round(float(np.percentile(samples, 50)), 3),
            "p95_ms": round(float(np.percentile(samples, 95)), 3),
            "p99_ms": round(float(np.percentile(samples, 99)), 3),
        })
    if errors:
        output["first_error"] = errors[0]
    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark local ShopIntent API HTTP latency")
    parser.add_argument("--url", default="http://127.0.0.1:8001/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--output", default="reports/performance/latest_api.json")
    args = parser.parse_args()
    if args.requests < 1 or any(n < 1 for n in args.workers):
        parser.error("requests and workers must be positive")
    warmup = [send_request(args.url) for _ in range(10)]
    if any(error for _, error in warmup):
        raise RuntimeError(f"API unavailable: {warmup[0][1]}")
    summary = [benchmark(args.url, workers, args.requests) for workers in args.workers]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"url": args.url, "results": summary}, indent=2), encoding="utf-8")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
