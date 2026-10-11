# Baseline HTTP benchmark (local Windows laptop)

Measurements supplied by the project developer; observed on localhost against the ShopIntent FastAPI prediction endpoint. The benchmark sent 50 measured HTTP requests per configuration. Hardware/CPU conditions, application startup flags, model-version details and repetitions were not documented.

| Clients | Success / requests | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 / 50 | 52.49 | 54.40 | 68.20 | 77.09 | 19.00 |
| 5 | 50 / 50 | 195.42 | 178.11 | 300.07 | 322.21 | 24.80 |

Five clients improved observed throughput by about **30.5%** but increased mean latency by approximately **3.7×**. This is exploratory evidence from a single local run, not a production SLA. Because each configuration used only 50 requests, p99 is especially unstable. The HTTP timing includes transport, FastAPI input processing, preprocessing, model inference and scheduling. The dedicated model benchmark separates pipeline-only processing for follow-up analysis.
