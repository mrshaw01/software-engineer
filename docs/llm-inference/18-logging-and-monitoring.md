# 18. Logging & Monitoring

This chapter defines a production-grade observability strategy for LLM inference systems. It covers metrics, logs, and distributed tracing; GPU/accelerator visibility; dashboards; SLOs and alerting; redaction and sampling policies; synthetic checks; and incident runbooks. Examples target Python (FastAPI) frontends with C++ backends, but the concepts generalize to TGI/vLLM/custom servers on NVIDIA/AMD hardware.

## 18.1 Objectives

- **Detect** regressions in latency, throughput, and error rates within minutes.
- **Explain** where time and resources are spent (router, tokenizer, prefill, decode, sampling, network).
- **Correlate** user-visible symptoms to component-level causes (e.g., batcher queue growth, KV cache thrashing, GPU OOM).
- **Protect** sensitive data with strict redaction and sampling.
- **Automate** SLO-based alerting and provide concise runbooks.

## 18.2 Observability Model

### Pillars and Scope

- **Metrics** (Prometheus/OpenMetrics): RED + LLM-specific (tokens/sec, batch efficiency, cache hit rate, decode speed).
- **Traces** (OpenTelemetry): request → router → worker → \[tokenize, prefill, decode-loop, sample, stream].
- **Logs** (structured JSON): request lifecycle + errors + rare forensic samples (redacted).

### Request Lifecycle (instrument every hop)

1. **Ingress/Router:** accepts requests, performs admission control, routes to a worker.
2. **Tokenizer/Preproc:** tokenization, prompt truncation, safety filters.
3. **Batcher/Scheduler:** queue and merge; prefill/decode separation.
4. **Model Execution:** prefill graph; decode iterations; sampler; KV cache ops.
5. **Egress/Streaming:** SSE/WebSocket chunking, backpressure.
6. **Postproc:** detokenization, formatting.

## 18.3 Metric Taxonomy & Naming

> Use low-cardinality labels. Avoid unbounded values (e.g., raw `user_id`). Prefer `tenant`, `model`, `route`, `worker_id`, `dtype`, `device_type`, and discretized `context_bucket`.

**RED/SLI Core**

- `inference_requests_total{model,route,code}` (counter)
- `inference_request_duration_ms_bucket{model,route}` (histogram)
- `inference_stream_resets_total{model,route}` (counter)
- `inference_active_streams{model}` (gauge)

**LLM-Specific**

- `tokens_generated_total{model,phase="decode"}` (counter)
- `tokens_input_total{model}` (counter)
- `tokens_per_second{model,phase}` (gauge) — exporter-calculated
- `batch_size_observed{model,phase}` (histogram)
- `batch_merge_wait_ms_bucket{model,phase}` (histogram)
- `kv_cache_hit_ratio{model}` (gauge)
- `kv_cache_bytes{model,device}` (gauge)
- `prefill_latency_ms_bucket{model}` (histogram)
- `decode_iter_latency_ms_bucket{model}` (histogram)
- `sampler_latency_ms_bucket{model}` (histogram)
- `tokenizer_latency_ms_bucket{model}` (histogram)
- `queue_time_ms_bucket{route}` (histogram)

**Scheduler/Speculative**

- `draft_accept_ratio{model}` (gauge)
- `speculative_verification_ms_bucket{model}` (histogram)

**GPU/Accelerator**

- `gpu_mem_bytes{device}` (gauge)
- `gpu_utilization_ratio{device}` (gauge)
- `gpu_mem_bw_ratio{device}` (gauge)
- `gpu_oom_events_total{device}` (counter)

**Cardinality Guards**

- Use `context_bucket` ∈ {`<=1k`,`1k-4k`,`4k-16k`,`>16k`}
- Use `tenant` or `project` when multi-tenant; never raw user IDs.

**Histogram Buckets (ms)**

- Request: `5,10,20,50,100,200,400,800,1600,3200,6400,12800`
- Prefill/Decode Iter: `0.2,0.5,1,2,4,8,16,32,64,128`
  Tune with exemplars to avoid bucket explosion.

## 18.4 Prometheus Export: Minimal Examples

### Python (FastAPI + prometheus_client)

```python
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

REQUESTS = Counter("inference_requests_total", "Requests", ["model","route","code"])
REQ_LAT = Histogram("inference_request_duration_ms", "Duration (ms)", ["model","route"],
                    buckets=[5,10,20,50,100,200,400,800,1600,3200,6400,12800])
TOKENS_OUT = Counter("tokens_generated_total", "Output tokens", ["model","phase"])
ACTIVE = Gauge("inference_active_streams", "Active streams", ["model"])

app = FastAPI()

@app.get("/metrics")
def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/chat/completions")
async def chat(req: Request):
    start = time.perf_counter()
    model = req.headers.get("x-model","gpt-oss-7m")
    route = "chat"
    ACTIVE.labels(model).inc()
    code = "200"
    try:
        # ... tokenize, prefill, decode ...
        generated = 128  # example
        TOKENS_OUT.labels(model, "decode").inc(generated)
        return {"tokens": generated}
    except Exception:
        code = "500"
        raise
    finally:
        ACTIVE.labels(model).dec()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        REQ_LAT.labels(model, route).observe(elapsed_ms)
        REQUESTS.labels(model, route, code).inc()
```

### C++ (prometheus-cpp, sketch)

```cpp
// Link prometheus-cpp pull exporter; expose /metrics on an HTTP endpoint.
auto& registry = *new prometheus::Registry();
auto& req_lat = prometheus::BuildHistogram()
  .Name("inference_request_duration_ms")
  .Help("Duration (ms)")
  .Register(registry)
  .Add({{"model","gpt-oss-7m"}, {"route","chat"}}, prometheus::Histogram::BucketBoundaries{5,10,20,50,100,200,400,800,1600,3200,6400,12800});

auto start = now_ms();
// ... inference ...
req_lat.Observe(now_ms() - start);
```

## 18.5 Distributed Tracing (OpenTelemetry)

**Span Model (avoid per-token spans; use events):**

- `inference.request` (root)

  - `router.admission`
  - `tokenize`
  - `batcher.queue` (attributes: `batch_id`, `phase`)
  - `prefill.graph.run`
  - `decode.loop` (add **events** per N tokens: `tokens_emitted=32`, add exemplars on `decode_iter_latency_ms`)
  - `sampler`
  - `stream.flush`
  - `postprocess`

**Required Attributes**

- `model`, `route`, `tenant`, `context_bucket`, `device_type`, `worker_id`, `batch_id`, `prefill_len`, `generated_len`, `slo_class` (e.g., `interactive`, `batch`).

### Python OTEL Init (OTLP over gRPC)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider(resource=Resource.create({"service.name":"inference-router"}))
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("inference.request", attributes={"model":"gpt-oss-7m"}):
    # child spans...
    pass
```

## 18.6 Structured Logging (JSON)

**Goals:** machine-parseable, low volume at steady state, rich on failure paths.
**Never log raw prompts or PII by default.** Opt-in, sampled, redacted, and access-controlled.

**Schema (example)**

```json
{
  "ts": "2025-08-18T04:11:12.345Z",
  "level": "INFO",
  "svc": "inference-worker",
  "req_id": "c1d9b...",
  "tenant": "acme",
  "model": "gpt-oss-20b",
  "route": "chat",
  "code": 200,
  "latency_ms": 842,
  "queue_ms": 31,
  "prefill_tokens": 512,
  "gen_tokens": 128,
  "batch_size": 8,
  "kv_cache_hit_ratio": 0.92,
  "device": "mi300-0",
  "error": null
}
```

**Python Logger (redaction + sampling)**

```python
import json, logging, os, random

class JsonHandler(logging.Handler):
    def emit(self, record):
        payload = record.__dict__.get("payload", {})
        # redact
        payload.pop("prompt", None)
        payload.pop("completion", None)
        line = { "ts": record.created, "level": record.levelname, **payload }
        print(json.dumps(line), flush=True)

log = logging.getLogger("obs")
log.setLevel(os.getenv("LOG_LEVEL","INFO"))
log.addHandler(JsonHandler())

def log_request(payload: dict):
    if random.random() < 0.01:  # 1% sample
        log.info("req", extra={"payload": payload})
```

## 18.7 GPU/Accelerator Visibility

- **NVIDIA:** deploy DCGM Exporter for SM utilization, mem BW, ECC, power, temperature.
- **AMD:** deploy ROCm SMI–based exporter; collect VRAM usage, temperature, power, clocks.
- **Correlate** device metrics with `worker_id` via labels; co-plot with `decode_iter_latency_ms` and queue sizes.
- Track `kv_cache_bytes` vs VRAM and **fragmentation**; alert on >85% sustained VRAM or frequent OOM.

## 18.8 Dashboards (Grafana)

**RED Overview**

- Panels: RPS, 5xx rate, P50/P95/P99 latency, active streams, queue time.

**LLM Performance**

- Tokens/sec (prefill/decode), batch size histogram, merge wait, context length distribution, KV hit ratio, speculative accept ratio.

**GPU/Worker**

- GPU util/mem/BW; worker request concurrency; VRAM headroom; OOM count.
- Overlay decode latency vs GPU util to detect under/over-utilization.

**Capacity & Sizing**

- Concurrency vs latency curves; admission control reject rate; scheduler saturation.

## 18.9 SLOs & Alerting

Define per **SLO class**:

- **Interactive:** `P95 request latency ≤ 2s`, `error rate ≤ 0.5%`.
- **Streaming:** first-token time ≤ 800ms P95; tokens/sec ≥ threshold.
- **Batch:** completion within window; throughput floor.

**Burn-Rate Alerts (multi-window)**

```yaml
# prometheus alerting rules
groups:
  - name: inference-slo
    rules:
      - alert: HighErrorBurnRate
        expr: |
          sum(rate(inference_requests_total{code=~"5.."}[5m]))
          / sum(rate(inference_requests_total[5m])) > 0.05
          and
          sum(rate(inference_requests_total{code=~"5.."}[1h]))
          / sum(rate(inference_requests_total[1h])) > 0.01
        for: 10m
        labels: { severity: page }
        annotations:
          summary: "Error budget burn"
          runbook: "https://…/runbooks/high-errors"

      - alert: LatencySLOViolated
        expr: histogram_quantile(0.95, sum by (le,model)(rate(inference_request_duration_ms_bucket[5m]))) > 2000
        for: 10m
        labels: { severity: page }
        annotations:
          summary: "P95 latency > 2s"
          runbook: "https://…/runbooks/high-latency"

      - alert: GPUOOMSpike
        expr: increase(gpu_oom_events_total[15m]) > 2
        labels: { severity: page }
        annotations:
          summary: "GPU OOM spike"
          runbook: "https://…/runbooks/gpu-oom"
```

## 18.10 Data Protection, Redaction, Sampling

- **Default: do not log** prompts/completions. Expose **masked hashes** for correlation.
- **PII policy:** automatic scrubbing of emails, phone numbers, SSNs; configurable allowlist for regulated tenants.
- **Sampling:** separate rates for success vs failure; elevate sampling on `code != 200`.
- **Retention:** short for verbose logs (e.g., 7d), longer for metrics/traces (e.g., 30–90d).
- **Access:** role-based, audited retrieval of sensitive samples.

## 18.11 Synthetic & Blackbox Monitoring

- **Ping** endpoints (health, `/v1/models`).
- **Canary prompts** per model/SLO class; assert first-token time, tokens/sec, and content guardrails.
- **Streaming validator:** ensure ordered, timely SSE chunks; alert on stall > N seconds.
- **Load probes** at off-peak to detect capacity regressions.

## 18.12 Common Pitfalls

- **High metric cardinality:** labels like `request_id`, free-form `error` strings, or raw `user_id`.
- **Per-token spans:** tracing overhead explodes; prefer events/exemplars.
- **Unbounded log volume:** missing sampling on success path.
- **GPU metrics mismatch:** device naming inconsistency across nodes.
- **Broken histograms:** non-monotonic bucket series or mis-tuned buckets leading to flat P95.

## 18.13 CI/CD & Testing for Observability

- **Unit tests:** exporters initialize; metric/label presence checks; histogram bucket coverage.
- **Golden traces:** snapshot critical span structure for a canary request.
- **Static analysis:** forbid disallowed labels; lint log schemas.
- **Load tests:** record baseline P95/P99, tokens/sec, batch efficiency; regress on PR.

## 18.14 Incident Runbooks (Templates)

**High P95 Latency**

1. Check RPS spike and admission rejects.
2. Inspect `queue_time_ms_bucket` and `batch_size_observed`.
3. Compare `gpu_utilization_ratio` vs `decode_iter_latency_ms`; if low util + high latency → CPU/IO bottleneck (tokenizer, network).
4. Verify KV hit ratio; if low, investigate context length shifts and eviction policy.
5. Roll back suspect deployment; scale out workers; tighten admission.

**Elevated 5xx**

1. Classify top error codes/messages.
2. Correlate with `gpu_oom_events_total` or worker restarts.
3. If model loading fails, drain router to healthy pools.
4. Increase sampling; capture exemplar traces.

**GPU OOM**

1. Examine `kv_cache_bytes` trend vs VRAM; look for fragmentation and batch size jumps.
2. Reduce max context or batch size; trigger cache compaction; recycle long-lived sessions.

## 18.15 Minimal Implementation Checklist

- [ ] Prometheus metrics exposed on every component.
- [ ] OpenTelemetry traces through router → worker with consistent `req_id`.
- [ ] JSON logs with redaction + sampling; log schema documented.
- [ ] Grafana dashboards: RED, LLM Perf, GPU/Worker.
- [ ] SLOs defined per class; burn-rate alerts configured.
- [ ] Synthetic canaries for each model/SLO class.
- [ ] Runbooks stored and linked from alerts.
- [ ] Periodic audits for label cardinality and bucket efficacy.

## 18.16 References (Implementation-Oriented)

- Prometheus/OpenMetrics client libraries (Python, C++, Go).
- OpenTelemetry SDKs and Collector (OTLP).
- GPU exporters: DCGM (NVIDIA), ROCm SMI–based exporters (AMD).
- Grafana: RED/USE method dashboards; histogram quantiles; exemplars.
