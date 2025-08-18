# Multi-Tenant Inference

This chapter covers design principles and practical implementation patterns for running multiple tenants (teams, customers, applications) on shared LLM inference infrastructure. The goals are: strong isolation, predictable performance, fair resource sharing, security, and operability at scale—without sacrificing throughput or latency.

## 22.1 Problem Definition and Goals

**Tenancy models**

- **Hard multi-tenancy (external customers):** strict isolation, audited access, per-tenant quotas and billing.
- **Soft multi-tenancy (internal teams/apps):** strong performance isolation and quotas; security constraints are lighter but still enforced.
- **Model sharing patterns:** (a) shared base model + per-tenant adapters (LoRA, prompts); (b) per-tenant model variants; (c) shared everything with per-request personalization.

**Key goals**

1. **Isolation:** prevent noisy-neighbor effects across compute, memory, storage, and network.
2. **Predictability:** SLOs per class of traffic (interactive vs. batch).
3. **Efficiency:** cross-tenant batching and cache sharing when safe.
4. **Security & compliance:** data boundaries, secrets handling, auditability.
5. **Cost fairness:** metering, quotas, and chargeback/showback.

## 22.2 Isolation Dimensions

- **Compute:** protect prefill/decoder kernel time; time-slicing and preemption at token boundaries.
- **Memory:** partition HBM/DDR for weights, KV cache, and temporary buffers; enforce per-tenant caps and fair reclaim.
- **I/O & network:** throttle per-tenant load (uploads, streaming responses), protect storage (model/object store).
- **Software isolation:** process/namespace boundaries; container sandboxing; per-tenant environment variables and secrets.
- **Security/privacy:** zeroization of freed buffers, per-tenant encryption keys, strict RBAC for controls and logs.

**Hardware notes**

- **NVIDIA:** MIG (A100/H100) for hard GPU partitioning; otherwise process-level isolation + MPS with care.
- **AMD:** SR-IOV/MxGPU where available; otherwise process isolation + driver context isolation.
- **When hardware partitioning is unavailable:** combine strict process/container isolation, cgroups, IOMMU, and software schedulers.

## 22.3 Resource Model

Track at least:

- **Prompt tokens** (prefill FLOPs dominate for long contexts).
- **Output tokens** (decode FLOPs dominate for long generations).
- **KV cache footprint** ≈ `n_layers × n_heads × head_dim × (prompt_tokens + generated_tokens) × bytes_per_element`.
- **Weights residency** (shared) vs. **adapters** (per tenant).
- **Bandwidth** (ingress/egress streaming).

Maintain per-tenant **resource meters** updated on admission and at each decode step.

## 22.4 Admission Control, Queues, and Batching

**Per-tenant rate limiting**

- **Token Bucket** per tenant and per priority class:

  - bucket unit: “scheduled tokens” (prefill + decode) or requests per second.
  - refill rate derives from contracted TPS/SLO.
  - burst to accommodate short spikes.

**Queue structure**

- **Per-tenant, per-class queues** (e.g., Interactive, Standard, Batch).
- **Global dispatcher** performs **Deficit Round Robin (DRR)** or **Weighted Fair Queuing (WFQ)** across queues to feed the batcher.

**Dynamic cross-tenant batching**

- Aggregate ready requests across tenants within a small batching window (e.g., 2–10 ms).
- Respect constraints: max batch size, shared model/adapters, token length similarity (to reduce padding waste), and SLO class.
- **Prefill/Decode separation**: build separate batches for prefill and decode paths to avoid head-of-line blocking.

**Age and starvation control**

- Apply **aging** (increase effective weight over time) or an **“oldest first” cap** to prevent starvation of low-weight tenants.

**Pseudocode: dispatcher + batcher (DRR)**

```python
class TenantQueue:
    def __init__(self, weight, class_name):
        self.q = deque()
        self.deficit = 0
        self.quantum = max(1, int(weight))
        self.class_name = class_name

class Dispatcher:
    def __init__(self, queues):
        self.queues = queues  # list[TenantQueue]

    def admit(self, req):
        req.estimate = estimate_tokens(req.prompt_len, req.max_new_tokens)
        if not token_bucket_allow(req.tenant, req.class_name, req.estimate):
            return REJECT_OR_503
        push_into_tenant_queue(req)

    def schedule(self):
        ready = []
        for tq in cycle(self.queues):
            tq.deficit += tq.quantum
            while tq.q and tq.q[0].estimate <= tq.deficit:
                r = tq.q.popleft()
                tq.deficit -= r.estimate
                ready.append(r)
                if batch_full(ready): break
            if batch_full(ready): break
        return form_batches(ready)  # split into prefill and decode microbatches
```

## 22.5 Preemption and Time-Slicing

- Preempt at **token boundaries** during decode; keep decode steps short (single-token).
- Use **CUDA/HIP Graphs** per phase to minimize overhead while permitting control between steps.
- For very long generations, enforce **per-request quantum** (max tokens per scheduling turn) to prevent monopolization.
- Enable **priority boosting** for near-deadline interactive requests.

## 22.6 Memory Management for Multi-Tenant

**Paged KV cache**

- Fixed-size blocks (e.g., 16–64 Ki tokens worth, or engine-specific block granularity).
- **Per-tenant KV pools** with hard/soft caps:

  - **Hard cap**: never exceed.
  - **Soft cap**: eligible for reclaim under pressure.

- **Eviction policy:** tenant-local LRU within soft region; global fair reclaim across tenants using proportional caps.

**Fragmentation control**

- Use contiguous virtual address spaces mapped to physical blocks.
- Background or opportunistic **compaction** when idle.
- Maintain **per-tenant reserve** to avoid priority inversions during spikes.

**Weights and adapters**

- Base weights loaded once and shared read-only across tenants.
- Per-tenant **LoRA/adapters/prompts**:

  - lazy-load on first request; keep in an **adapter cache** with LRU + pin option for premium tenants.
  - apply **quorum warmup** (preload top-N adapters on scale-up).

**Zeroization and safety**

- Zero/free scratch buffers and evicted KV pages before reuse when crossing tenant boundaries.
- Validate tensor shapes and lengths defensively at API boundaries.

## 22.7 QoS Classes and SLOs

Define classes such as:

- **Interactive (P0):** tight p95 latency (e.g., 300–800 ms for first token), strict jitter control.
- **Standard (P1):** moderate latency, higher throughput.
- **Batch (P2):** best-effort throughput, relaxed latency.

**SLO metrics**

- Time-to-First-Token (TTFT), Tokens-per-Second (TPS), p50/p95/p99 end-to-end latency, availability, error rate.
- **Error budgets** per class; automatic **degrading strategies**:

  - shrink `max_new_tokens`, switch to a smaller model, or drop to non-streaming.

**Back-of-envelope sizing**

For steady state, approximate per-tenant token budget:

```
tenant_tps <= (gpu_tps_total * tenant_weight / sum_weights) * efficiency
```

Where `gpu_tps_total` depends on model size, kernel efficiency, and batching.

## 22.8 Overload and Noisy-Neighbor Protection

- **Three layers of defense:**

  1. **Client-side limits** (SDK backpressure).
  2. **Ingress rate limit** (per API key/tenant).
  3. **Scheduler admission control** (token bucket + queue caps).

- **Global surge control:** if GPU utilization or queue depth crosses thresholds, tighten admission (reduce bucket refill), shorten batching window for interactive class, and reject batch traffic first.
- **Per-tenant circuit breaker:** temporarily shed load from misbehaving tenants.

## 22.9 Autoscaling and Placement

**Horizontal scaling**

- Scale **router/dispatcher** and **workers** independently.
- Predictive scaling from moving averages of request rate and token counts; keep “adapter warmers” to mitigate cold-start.

**Vertical/partitioning**

- If available, use MIG/SR-IOV partitions to enforce hard caps.
- Otherwise, **bin-pack** tenants to workers using:

  - weight-based demand (tokens/sec),
  - memory footprint (KV + adapters),
  - hotness (adapter switching cost).

**Model residency**

- Keep base weights resident on all workers of a pool.
- Place adapters by popularity; replicate to N workers to cap cross-machine loads.

## 22.10 Security, Compliance, and Privacy

- **Transport:** TLS; **storage:** encrypt model and logs; integrate with KMS.
- **Secrets:** per-tenant secrets vault; inject via environment or sidecar with short-lived tokens.
- **RBAC & audit:** per-tenant scopes for metrics/log access; immutable audit logs.
- **Data minimization:** configurable retention; PII scrubbing before persistence.
- **Memory hygiene:** zero KV/temps on eviction; forbid cross-tenant pointer reuse; fuzz/ASAN builds in CI for safety.

## 22.11 Observability and Metering

**Metrics (illustrative)**

- `inference_requests_total{tenant, class}`
- `prompt_tokens_total{tenant}`, `completion_tokens_total{tenant}`
- `ttft_seconds_bucket{tenant, class}`, `tps_tokens_per_second{tenant}`
- `kv_bytes_in_use{tenant}`, `kv_evictions_total{tenant}`
- `adapter_cache_hits_total{tenant, adapter}`

**Tracing**

- Propagate `trace_id`, `tenant_id`, `request_class`.
- Span boundaries at: ingress, admission, batch build, prefill run, decode step, stream write.

**Noisy-neighbor detectors**

- Sudden KV growth rate per tenant.
- High adapter thrash (loads/min).
- Queue wait p95 skew by tenant vs. pool median.

## 22.12 API and Policy Surface

**Per-tenant config (example)**

```yaml
tenant_id: "acme"
classes:
  interactive:
    weight: 8
    token_bucket:
      rate_tps: 600
      burst_tps: 900
    slo:
      ttft_p95_ms: 500
      availability: "99.9%"
  batch:
    weight: 2
    token_bucket:
      rate_tps: 200
      burst_tps: 400
quotas:
  kv_hbm_mb: 12288 # soft
  kv_hbm_mb_hard: 16384
adapters:
  pin: ["lora-v5", "lora-v6"]
security:
  can_stream: true
  data_retention_days: 7
```

**Request headers**

- `X-Tenant-ID`, `X-Request-Class`, optional `X-Adapter-ID`, `X-Priority`.

## 22.13 Implementation Notes (vLLM/TGI-like Engines)

- Use **prefill/decode queues** with per-tenant weights. The decode loop naturally supports **token-boundary preemption**.
- Maintain a **per-tenant KV allocator** atop the engine’s paged KV (e.g., “blocks”); enforce per-tenant soft/hard caps and export metrics.
- Extend batcher to **mix tenants** under the same model/adapters with length-aware grouping.
- Adapter manager: **async load** + **reference counting**; integrate with eviction policy (LRU + pinning).

## 22.14 Cost Accounting

**Per-request estimated FLOPs**

- Prefill ≈ `2 * L * H * S` (order-of-magnitude), where `L` layers, hidden size `H`, sequence length `S`.
- Decode ≈ similar per-token cost; multiply by generated tokens `G`.
- Billable unit: prompt tokens, completion tokens, or normalized FLOPs.
- Track **cache hit credits** (KV reuse reduces effective cost).

## 22.15 Testing and Reliability

- **Load shapes:** steady, bursty, diurnal; synthetic tenants with different mixes and SLOs.
- **Chaos testing:** kill a worker, drop adapter cache, spike one tenant; verify SLO impact is contained.
- **Priority inversion tests:** ensure interactive traffic maintains TTFT when batch load surges.
- **Failover:** router retries with idempotency keys; preserve state for streaming reconnection where feasible.

## 22.16 Operational Runbook (Checklist)

1. Define tenants, classes, weights, token-bucket rates.
2. Set KV caps per tenant; size adapter cache and pin critical adapters.
3. Configure dispatcher (DRR/WFQ), batching window, and decode quantum.
4. Enable overload controls and circuit breakers.
5. Ship per-tenant metrics and traces; build dashboards per class.
6. Establish SLOs and error budgets; wire automated degrade policies.
7. Validate security posture: secrets, RBAC, encryption, memory zeroization.
8. Capacity plan by token budgets; set autoscaling rules.
9. Run chaos/load tests; iterate on policies.

## 22.17 Reference Snippets

**Token bucket gate**

```python
class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate
        self.capacity = burst
        self.tokens = burst
        self.ts = now()

    def allow(self, cost):
        self.refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def refill(self):
        t = now()
        self.tokens = min(self.capacity, self.tokens + (t - self.ts) * self.rate)
        self.ts = t
```

**Decode quantum enforcement**

```python
MAX_TOKENS_PER_TURN = 16  # prevents monopolization

def decode_turn(request):
    to_generate = min(request.remaining, MAX_TOKENS_PER_TURN)
    for _ in range(to_generate):
        step_decode_one_token(request)
        if near_deadline(request): break
```

**Tenant-aware KV eviction (conceptual)**

```python
def reclaim_kv(required_bytes):
    # First, try tenant-local LRU within soft quotas
    for tenant in tenants_sorted_by_pressure():
        freed = evict_from_tenant_lru(tenant, required_bytes)
        if freed >= required_bytes: return True
    # Then, proportional fair reclaim across tenants (respect hard caps)
    for tenant in tenants_by_share_deficit():
        freed = evict_from_tenant_soft_region(tenant, required_bytes)
        if freed >= required_bytes: return True
    return False
```

## 22.18 Summary

A robust multi-tenant inference platform combines (1) **policy**—weights, quotas, and SLOs, (2) **mechanisms**—token-bucket admission, fair queuing, cross-tenant batching, per-tenant KV management, and token-boundary preemption, and (3) **operations**—observability, autoscaling, and security. Execute on all three to deliver predictable, efficient, and safe performance for diverse tenants.
