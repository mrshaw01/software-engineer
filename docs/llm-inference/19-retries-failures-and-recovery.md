# Retries, Failures, and Recovery

Reliable LLM serving requires explicit strategies for handling transient faults, overload, device failures, and partial progress during streaming. This section provides actionable policies, reference implementations, and operational playbooks for robust retries and fast recovery—across routers, model workers, and clients.

## 19.1 Design Tenets

1. **Idempotency first:** Make write-like operations (generation that touches quotas/billing) safely retryable via idempotency keys.
2. **Bounded retries:** Use capped exponential backoff with jitter and **retry budgets** (not “infinite patience”).
3. **Deadline propagation:** Each hop respects a **timeout budget** (e.g., `grpc-timeout`, `X-Request-Timeout`) to prevent retries that can’t finish.
4. **Circuit breaking & backpressure:** Fail fast under overload to protect tail latency and avoid cascading failures.
5. **Graceful degradation:** Prefer partial service (shorter max tokens, smaller batch, fallback precision/model) over hard failure.
6. **Resumability for streams:** Make streaming **resumable** via generation IDs and sequence numbers.
7. **Fast-path recovery:** Optimize for **hot restart** (cache or state snapshot) rather than cold restart.

## 19.2 Failure Modes in LLM Serving

| Layer            | Common Failure                        | Typical Cause                                   | Detect/Signal                            |
| ---------------- | ------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| Client ↔ Router | 4xx/5xx, timeouts, dropped TCP/SSE    | Rate limits, proxy resets, NAT timeouts         | HTTP status, SSE disconnect, gRPC status |
| Router           | Queue runaway, high latency, OOM      | Over-admission, large prompts, burst traffic    | Queue depth, p95/99, memory alerts       |
| Worker (GPU/NPU) | Device OOM, kernel fault, memory leak | Oversized batch/KV, fragmentation, buggy kernel | HIP/CUDA/Xid, ROCm/HSA error codes       |
| Collectives      | NCCL/RCCl timeout/abort               | Network blip, rank desync                       | Communicator error, health probe         |
| Storage/Cache    | Miss/eviction, slow I/O               | Under-provisioned cache, churn                  | Hit ratio, I/O latency                   |
| Streaming        | Mid-stream disconnect                 | Client network, proxy idle timeout              | Broken pipe, half-closed socket          |

## 19.3 Error Classification & Retry Policy

Map transport and application errors to **retryability** and **client/server action**.

| Error Class                    | Examples                             |                  Client Retry? | Server Action                     | Notes                                          |
| ------------------------------ | ------------------------------------ | -----------------------------: | --------------------------------- | ---------------------------------------------- |
| **429 / RateLimited**          | Admission control, quota             |                            Yes | Return `Retry-After`              | Jittered backoff; prefer token bucket feedback |
| **503 / Unavailable**          | Rolling restarts, temporary overload |                            Yes | Shed/queue; trip breaker          | Hedge only if idempotent                       |
| **500 / Internal** (transient) | Router restart, worker crash         |                  Yes (limited) | Mark node unhealthy; restart pool | Use idempotency key                            |
| **Timeout**                    | Deadline exceeded                    | Yes (within original deadline) | Release resources                 | Propagate remaining budget                     |
| **Device OOM**                 | Large batch/KV                       |          Maybe (after degrade) | Reduce batch/max tokens; re-admit | Retry with smaller request                     |
| **Client Error 4xx**           | Invalid params, auth                 |                             No | —                                 | Surface clear message                          |
| **Deterministic Model Error**  | Prompt too long, bad params          |                             No | —                                 | Offer corrective suggestion                    |

**Rule of thumb:** Retry only **safe** and **likely transient** failures, and only within the **original deadline**.

## 19.4 Idempotency & Exactly-Once Semantics

Support **idempotency keys** to make request/retry safe:

- **Header:** `Idempotency-Key: <uuid>` plus a **body hash** (optional) to prevent key reuse with different payloads.
- **Server store:** Cache `(key → terminal result or failure)` for a TTL (e.g., 24–72h).
- **Billing/quotas:** Charge once per idempotency key on the first successful completion.
- **Streaming:** Attach a **generation_id** and **sequence index**; if retried, the server can replay from the last acknowledged token.

**OpenAI-style API tip:** Support `X-Request-ID` for tracing and a distinct `Idempotency-Key` for semantics.

## 19.5 Deadlines, Timeouts, and Budgets

- **Client** sets a **total deadline** (e.g., 30s) and **retries only within** that budget.
- **Hop budgets:** Each service subtracts its spent time and forwards the remainder (e.g., `grpc-timeout: 12S`).
- **Per-phase budgets:**

  - **Prefill** (context) has a tighter timeout than **Decode** (streaming).
  - Avoid spending >50% of the total budget in prefill.

**Admission hint:** If the remaining budget < minimum service time estimate, **fail fast** with a helpful error.

## 19.6 Backoff, Jitter, and Hedging

Use **capped exponential backoff with jitter**:

```python
# Pseudocode
base = 0.1  # seconds
cap = 2.0   # seconds
retries = 4
sleep = base
for attempt in range(retries):
    try:
        return call()
    except Retryable as e:
        # Full jitter
        delay = random.uniform(0, min(cap, sleep))
        time.sleep(delay)
        sleep *= 2
raise FinalError
```

- **Hedged requests** (duplicate to a second worker) only for **idempotent** reads or generation with idempotency key; cancel the slower one.
- Enforce a **retry budget**, e.g., total sleep ≤ 40% of client deadline.

## 19.7 Circuit Breakers & Load Shedding

- **Breaker states:** _Closed_ → _Open_ (on error rate/latency spikes) → _Half-Open_ (probe).
- **Trip conditions (example):** error rate > 15% over 30s, or p99 > SLA × 2, or queue depth > threshold.
- **Shed tiers (brownouts):**

  1. Reduce **max_new_tokens** / truncate context to sliding window.
  2. Lower precision or switch to a smaller model.
  3. Disable non-critical features (tools, function-calling).
  4. Reject new requests with informative 503 + `Retry-After`.

## 19.8 Resilience for Streaming (SSE/WebSocket/gRPC)

**Resumable streams**:

- Server emits: `(generation_id, token_index, token_text)`.
- Client sends on resume: `(generation_id, last_received_index)`.
- For **SSE**, use `Last-Event-ID`; for **gRPC**, add a resume request with the same identifiers.
- Server replays tokens from `last_received_index + 1` and continues decoding if the session is alive.

**Periodic checkpoints**:

- Every _K_ tokens (e.g., 16–32), persist a small **decode state checkpoint**: sampling RNG seed/counter, logits mask state, penalty state, and **KV-cache handle** if supported.

## 19.9 Worker Recovery & KV Cache Strategies

1. **Local crash mid-decode:**

   - Reconstruct state from `(prompt, generated_tokens)` and **recompute tail** (e.g., last 64 tokens) to restore KV; then resume.
   - If **paged KV cache** supports spill to host/NVMe, reload the last checkpointed pages.

2. **Device OOM / Fragmentation:**

   - Retry with **smaller batch**, shorter context, or **quantized** KV (e.g., Q8) for decode.
   - If attention supports **sliding window**, cap attention to window size on retry.

3. **Distributed decode (multi-GPU/NPU):**

   - On **RCCl/NCCL** error: abort communicator, **tear down the group**, and re-form (ranks rejoin).
   - Resume from the last **synchronized checkpoint** (token boundary).

4. **Speculative decoding:**

   - If **draft/target desync** or target crash: **rollback** to last verified token and **degrade** to base decoding; optionally disable draft on subsequent retry.

## 19.10 Router Strategies: Admission, Queues, and Cancellation

- **Predictive admission:** Estimate prefill FLOPs and KV memory; **reject early** if unmet.
- **Queue control:** Enforce per-tenant quotas and **max wait**; if queue time > budget, fail fast.
- **Cancellations:**

  - On client disconnect or timeout budget expiry, **propagate cancel** to workers.
  - **Garbage collect** orphaned KV within a TTL.

- **Sticky sessions:** Keep a session on the same worker for KV locality; if the worker fails, **rehydrate** or restart from last checkpoint.

## 19.11 Policy Matrix (Server)

| Condition                   | Automated Response                                                      | Retry Hint                   |
| --------------------------- | ----------------------------------------------------------------------- | ---------------------------- |
| Error rate > 10% and rising | Trip **service-level breaker**, shed new prefill, allow existing decode | Clients back off (429/503)   |
| p99 latency > SLA × 2       | Enable **brownout tier 1** (reduce tokens)                              | Clients keep deadline budget |
| GPU memory < 5% free        | Reduce per-request max tokens; block large prompts                      | Retry with smaller request   |
| KV hit rate < 60%           | Increase cache size or eviction aggressiveness                          | —                            |
| Communicator faults         | Reset group; quarantine bad node                                        | Safe to retry with same key  |

## 19.12 Client-Side Reference (HTTP/JSON)

### Request headers

- `Idempotency-Key: <uuid>`
- `X-Request-Timeout: 30s` (or `grpc-timeout`)
- `Accept: text/event-stream` (for SSE)

### Streaming resume (SSE)

- Initial: `POST /v1/chat/completions` → events include `generation_id`, `index`
- Resume: `GET /v1/streams/{generation_id}` with `Last-Event-ID: <index>`

### Retry loop (pseudo)

```python
def generate_with_retry(payload, timeout_s=30, max_attempts=4):
    deadline = time.monotonic() + timeout_s
    key = uuid4().hex
    backoff = 0.1
    for attempt in range(max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("client deadline exceeded")
        try:
            return post_json("/v1/chat/completions",
                             payload,
                             headers={"Idempotency-Key": key,
                                      "X-Request-Timeout": f"{int(remaining)}s"},
                             timeout=remaining)
        except RetryableHTTP as e:
            sleep = random.uniform(0, min(2.0, backoff))
            time.sleep(sleep)
            backoff *= 2
    raise FinalError("exhausted retries")
```

## 19.13 Persistence & Exactly-Once Accounting

- Persist **final result** keyed by `(tenant_id, idempotency_key)` with:

  - Request hash, sampling params, model version
  - Usage (tokens in/out), cost, and **checksum of output**

- On retry with same key:

  - If **in-flight**, return **202 Accepted** with stream endpoint.
  - If **completed**, return the **same result** and **same usage**.
  - If **failed deterministically**, return the same error without re-executing.

## 19.14 Chaos & Failure Injection

Adopt continuous **chaos testing** to validate policies:

- **Router restarts** during active streaming.
- **Worker kill -9** mid-decode; verify resume from checkpoint.
- **Network partitions** causing RCCl/NCCL timeouts.
- **Device OOM** by oversizing batch/prompt.
- **Cache eviction storms** simulating burst traffic.

**Success criteria:** bounded error rates, preserved SLAs, correct billing exactly-once, and state cleanup after aborts.

## 19.15 Operational Playbooks

1. **Spike Overload**

   - Trigger breaker; enable brownout tier 1; raise queue thresholds.
   - Notify: SRE on-call; tenants via status page with ETA and mitigation.

2. **Hot Node with Elevated OOM**

   - Drain node; defragment KV; reduce max batch for the AZ; re-admit gradually.

3. **Collective Failures**

   - Reset communicator; reroute new sessions to healthy pool; retry existing ones with idempotency keys.

4. **Regional Degradation**

   - Shift traffic via router policy; enable **read-replica** for embeddings; publish `Retry-After`.

## 19.16 Implementation Notes (CUDA/HIP & AMD MI-series)

- On ROCm/HIP errors, **destroy and recreate** the context/stream; **discard** all buffers tied to the failed stream.
- Pre-allocate **paged KV** slabs with guard pages; on OOM, switch to **quantized KV** or **sliding window**.
- Keep **checkpoint size** small: serialize only RNG counter, penalty state, and references to KV pages (not full tensors).

## 19.17 API Contract Checklist

- [ ] Supports `Idempotency-Key` and returns deterministic results for the same key.
- [ ] Exposes `generation_id` and token indices in streams.
- [ ] Returns `Retry-After` for 429/503 with reasonable bounds.
- [ ] Emits machine-readable error codes (`code`, `retryable`, `hint`).
- [ ] Documents **max tokens after brownout** and **fallback model** behavior.
- [ ] Provides **usage** in final and resumed responses.

## 19.18 Minimal Server Hooks (pseudo)

```cpp
// On request admission
if (!has_budget(req.deadline) || queue_full()) {
  return http_429_retry_after(budget_hint());
}

// Idempotency lookup
auto key = req.idempotency_key;
if (auto cached = store.find(key)) return cached;

// Execute with guard
Result r;
try {
  r = run_generation(req, deadline);
  store.persist(key, r); // exactly-once
  return r;
} catch (RetryableError& e) {
  mark_node_health(e);
  throw;
} catch (DeterministicError& e) {
  store.persist(key, e.to_result());
  throw;
}
```

## 19.19 Measuring Success

- **SLOs:** p95/p99 latency per endpoint, success rate excluding client errors.
- **Retry effectiveness:** fraction of successes **after** at least one retry.
- **Wasted work:** canceled/aborted tokens vs. emitted tokens.
- **False retries:** retries on non-retryable errors (should trend to zero).
- **Recovery time:** mean time to healthy after chaos events.

## 19.20 Summary

A robust LLM serving stack treats retries as a **disciplined protocol**, not a guess: strict idempotency, bounded backoff, deadline-aware routing, circuit breaking, resumable streaming, and fast KV/state recovery. When combined with chaos testing and brownout strategies, these practices keep latency predictable and availability high—even under failure.
