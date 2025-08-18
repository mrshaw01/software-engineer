# Async/Streaming Inference

This section describes how to deliver partial model outputs _as they are generated_, with a focus on protocol choices (SSE, WebSocket, gRPC), server design (prefill/decode lifecycle, batching, cancellation), backpressure, observability, and production hardening. Examples are provided to implement token streaming with clear, testable contracts.

## 1) Objectives & Terminology

**Objectives**

- Minimize perceived latency (time to first token).
- Keep tokens flowing smoothly (low inter-token latency).
- Maintain high throughput under load via dynamic batching and fair scheduling.
- Provide robust cancellation, retries, and resilient connections through proxies and CDNs.

**Key metrics**

- **TTFT (Time To First Token):** wall time from request receipt to first streamed token.
- **ITL (Inter-Token Latency):** distribution and p95/p99 between consecutive tokens.
- **TPOT (Tokens Per Output Time):** output tokens / second per request and per node.
- **Stream Stability:** resets, disconnects, backoff/retry success rates.

## 2) Protocol Options (SSE vs WebSocket vs gRPC)

| Aspect                   | Server-Sent Events (SSE)               | WebSocket                            | gRPC Streaming                   |
| ------------------------ | -------------------------------------- | ------------------------------------ | -------------------------------- |
| Transport                | HTTP/1.1, unidirectional server→client | Full-duplex over a single TCP        | HTTP/2 streams                   |
| Ease through proxies/CDN | Excellent (with correct headers)       | Sometimes restricted by corp proxies | Good on modern infra             |
| Backpressure             | Limited (TCP-level only)               | Application-level possible           | Built-in flow control            |
| Client simplicity        | Very simple (EventSource, `curl`)      | Moderate (WS libs)                   | Requires gRPC toolchain          |
| Framing                  | `text/event-stream` lines              | App-defined frames                   | Protobuf messages                |
| Best use                 | OpenAI-style token deltas              | Interactive apps/tools               | Strongly-typed internal services |

**Guidance**

- Use **SSE** for public, HTTP-native “OpenAI-compatible” APIs.
- Use **WebSocket** for interactive tool UIs needing client→server control mid-stream.
- Use **gRPC** for internal, typed microservices (router↔worker, worker↔scheduler).

## 3) Wire-Level Details (SSE)

**Required headers**

```
Content-Type: text/event-stream
Cache-Control: no-cache, no-transform
Connection: keep-alive
X-Accel-Buffering: no        # nginx
```

**Framing**

- Lines starting with `data: ` carry payload (usually JSON).
- Separate events with a blank line (`\n\n`).
- Send periodic **heartbeats** to keep intermediaries from closing idle connections:

  - Comment lines: `:\n\n` (preferred) or minimal `data: {}\n\n`.

**Proxy/CDN considerations**

- **Disable buffering** (e.g., `proxy_buffering off;` in nginx) for streaming.
- Enforce **HTTP/1.1 keep-alive** and do not enable gzip for streams by default.
- Set conservative **idle timeouts** on LB/CDN (≥ 60s) and send heartbeats < timeout/2.

**Event contract (OpenAI-style example)**

- Token deltas:

  ```json
  {
    "id": "...",
    "object": "chat.completion.chunk",
    "choices": [
      { "delta": { "role": "assistant", "content": "Hel" }, "index": 0 }
    ]
  }
  ```

- Final message (includes usage, finish reason), then a literal `[DONE]` sentinel:

  ```
  data: {"id":"...","object":"chat.completion","choices":[{"finish_reason":"stop","index":0}]}

  data: [DONE]

  ```

## 4) Server Design for Streaming

### 4.1 Prefill/Decode Lifecycle

1. **Prefill**: embed and run the prompt once; populate KV cache.
2. **Decode loop**: one (or a few) tokens per step, reusing KV cache.
3. **Emit**: push deltas as soon as each token is available.

### 4.2 Dynamic Batching While Streaming

- Batch requests by **phase** (prefill vs decode).
- Use micro-batching windows (e.g., 1–5 ms) to aggregate concurrent decodes.
- Merge variable-length sequences with **paged/prefixed attention** to avoid padding waste.
- Maintain **per-request ordering** when de-batching outputs.

### 4.3 Flow Control & Backpressure

- SSE relies on TCP; implement **application-level pacing**:

  - Per-stream in-memory queue with soft/hard limits (e.g., 256/1024 chunks).
  - If client is slow, **coalesce** multiple tokens into one chunk.
  - If queue hits hard limit, **drop connection** gracefully with a retry-able error.

- For gRPC/WS, use **credit-based** token windows.

### 4.4 Cancellation & Deadlines

- Propagate client disconnects to cancel GPU work promptly.
- Support `timeout_ms` per request; surface `finish_reason="length"` on timeout.
- Make the API **idempotent** using a client-supplied `request_id` to deduplicate on reconnect.

### 4.5 Error Semantics

- Non-retryable: `400` (validation), `403` (auth), `422` (model refuses).
- Retryable: `408/499` (client cancel), `429` (throttle), `502/503/504` (upstream).
- Stream errors should send a structured error event when possible before closing.

## 5) Minimal SSE Implementation (Python, FastAPI/Uvicorn)

> Contract: emits OpenAI-style chat deltas, heartbeats every 15s, `[DONE]` at end.
> Notes: disable gzip, ensure `proxy_buffering off;`, and use HTTP/1.1.

```python
# app.py
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio
import json
import time

app = FastAPI()

async def generate_tokens(prompt: str):
    # TODO: integrate with your inference engine
    # Simulate TTFT
    await asyncio.sleep(0.05)
    for tok in ["Hel", "lo", ", ", "wor", "ld", "!"]:
        yield tok
        await asyncio.sleep(0.02)  # simulate inter-token latency

async def sse_stream(prompt: str, request: Request) -> AsyncGenerator[bytes, None]:
    start = time.time()
    # First delta (optionally include role)
    first_delta = {"id": "req-123", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    yield f"data: {json.dumps(first_delta)}\n\n".encode()

    last_hb = time.time()
    async for tok in generate_tokens(prompt):
        if await request.is_disconnected():
            # Stop GPU work here if applicable
            break
        payload = {"id": "req-123", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": {"content": tok}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()

        # Heartbeat every 15s to keep intermediaries alive
        now = time.time()
        if now - last_hb > 15:
            yield b":\n\n"  # comment heartbeat
            last_hb = now

    # Final message with finish_reason and usage (if available)
    final = {"id": "req-123", "object": "chat.completion",
             "choices": [{"index": 0, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("messages", [{"content": ""}])[-1].get("content", "")
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(sse_stream(prompt, request), headers=headers)
```

**Run**

```bash
uvicorn app:chat --factory --host 0.0.0.0 --port 8000 --http h11
# or uvicorn app:app --host 0.0.0.0 --port 8000 --http h11
```

**Test**

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss", "stream": true,
       "messages":[{"role":"user","content":"Say hello"}]}'
```

## 6) WebSocket Outline (Bidirectional Control)

**When to use**

- Tools/plugins sending function calls and receiving tool outputs mid-stream.
- Clients that need to **pause/resume**, adjust sampling parameters, or request **tool call cancellations**.

**High-level flow**

1. Client connects to `/v1/realtime` (authenticated).
2. Client sends `start` with prompt and parameters.
3. Server streams `token` events; client may send `pause`, `resume`, or `cancel`.
4. Server sends `final` with usage and closes (or stays open for next turns).

**Message shape (example)**

```json
// client → server
{ "type": "start", "request_id": "abc", "messages": [...], "params": { "temperature": 0.7 } }

// server → client
{ "type": "token", "request_id": "abc", "delta": "Hel" }
{ "type": "final", "request_id": "abc", "finish_reason": "stop", "usage": {...} }
```

## 7) gRPC Streaming (Internal Service-to-Service)

**Proto sketch**

```proto
syntax = "proto3";
package llm.v1;

message StartRequest {
  string request_id = 1;
  repeated string messages = 2;
  float temperature = 3;
  int32 max_tokens = 4;
  int32 timeout_ms = 5;
}

message StreamChunk {
  string request_id = 1;
  oneof payload {
    TokenDelta delta = 2;
    FinalResult final = 3;
    Error error = 4;
    Heartbeat hb = 5;
  }
}

message TokenDelta { string text = 1; }
message FinalResult { string finish_reason = 1; int32 output_tokens = 2; }
message Error { int32 code = 1; string message = 2; }
message Heartbeat { int64 ts_unix_ms = 1; }

service Inference {
  rpc Stream (StartRequest) returns (stream StreamChunk);
}
```

**Notes**

- Rely on HTTP/2 flow control.
- Use deadlines and cancellations from the client stub to immediately free compute.
- Prefer **small** protobuf messages for token deltas to reduce latency.

## 8) Scheduling, Batching, and QoS While Streaming

- **Two-queue model**: separate **prefill** and **decode** pools; prefill is bursty and heavier, decode is steady and light.
- **Fairness**: round-robin across streams in the decode micro-batch builder to avoid starvation of long tails.
- **Priorities**: apply per-tenant or per-SKU priorities; cap max parallel prefill per tenant to protect TTFT.
- **Speculative decoding**: if available, stream from the verifier as tokens clear; rollback on rejection is invisible to clients.
- **Preemption**: allow high-priority streams to preempt low-priority prefill; never preempt an in-flight decode kernel.

## 9) Observability & SLOs

**Metrics (Prometheus-friendly)**

- `stream_ttft_seconds{model,route}`
- `stream_tokens_per_second{model,phase="decode"}`
- `stream_inter_token_p95_seconds{model}`
- `stream_disconnects_total{cause}`
- `stream_retries_total{reason}`
- `scheduler_batch_size{phase}` and `scheduler_wait_time_seconds{phase}`

**Tracing**

- Span: `prefill`, `decode_step[n]`, `emit_chunk[m]`.
- Attributes: `batch_size`, `seq_len`, `kv_cache_bytes`, `kernel_ms`.

**Logs**

- One structured log per request with `request_id`, `ttft_ms`, `tokens_out`, `finish_reason`, `retries`.

**SLO examples**

- p95 TTFT ≤ 150 ms (cached prompt) / ≤ 800 ms (cold).
- p95 inter-token latency ≤ 35 ms for models ≤ 13B at seq≤256 on target HW.
- Error budget on resets ≤ 0.1%.

## 10) Production Hardening Checklist

- **API stability:** documented event schema, `[DONE]` sentinel, explicit `finish_reason`.
- **Auth & quotas:** bearer tokens, per-tenant rate limits, concurrent stream caps.
- **Load shedding:** 429 with `retry_after`, jittered backoff guidance to clients.
- **Time limits:** hard per-request `max_duration_ms`.
- **Safety filters:** streaming content moderation with early termination.
- **Resource guards:** KV cache eviction policy, per-request memory budgets.
- **Proxy configs:** disable buffering; increase idle timeouts; HTTP/1.1 keep-alive.
- **Compression:** off by default; only enable if intermediaries preserve chunk flush.
- **Idempotency:** accept `request_id` to dedupe retried starts.
- **Reconnection story (optional):** short-lived resumable window with token offset.

## 11) Client Snippets

**Browser (SSE)**

```html
<script>
  const es = new EventSource("/v1/chat/completions?stream=true"); // if GET flavor
  es.onmessage = (e) => {
    if (e.data === "[DONE]") {
      es.close();
      return;
    }
    const msg = JSON.parse(e.data);
    const delta = msg.choices?.[0]?.delta?.content || "";
    document.getElementById("out").textContent += delta;
  };
  es.onerror = () => es.close();
</script>
<pre id="out"></pre>
```

**Python (requests, SSE)**

```python
import requests, json
r = requests.post("http://localhost:8000/v1/chat/completions",
                  json={"stream": True, "messages":[{"role":"user","content":"Hi"}]},
                  stream=True)
for line in r.iter_lines(decode_unicode=True):
    if not line: continue
    if line.startswith("data: "):
        data = line[6:]
        if data == "[DONE]": break
        chunk = json.loads(data)
        print(chunk["choices"][0]["delta"].get("content",""), end="", flush=True)
```

## 12) Testing & Tuning

- **Latency smoke test**

  ```bash
  time curl -N -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"stream": true, "messages":[{"role":"user","content":"Count to five"}]}'
  ```

- **Throughput test**

  - Run 20–100 concurrent streams (e.g., `oha`, `wrk2`) with think-time.
  - Validate p95 TTFT/ITL and absence of proxy buffer stalls.

- **Proxy validation**

  - Verify heartbeats bypass buffering.
  - Confirm no gzip/content-length is injected.

## 13) Reference Event Schema (SSE)

- **Delta chunk**

  ```json
  {
    "id": "string",
    "object": "chat.completion.chunk",
    "created": 1734550000,
    "model": "gpt-oss-7m",
    "choices": [
      {
        "index": 0,
        "delta": { "role": "assistant", "content": "..." },
        "logprobs": null
      }
    ]
  }
  ```

- **Final**

  ```json
  {
    "id": "string",
    "object": "chat.completion",
    "created": 1734550005,
    "model": "gpt-oss-7m",
    "choices": [{ "index": 0, "finish_reason": "stop" }],
    "usage": {
      "prompt_tokens": 42,
      "completion_tokens": 128,
      "total_tokens": 170
    }
  }
  ```

- **Terminal sentinel**

  ```
  [DONE]
  ```

## 14) Common Pitfalls

- Missing `X-Accel-Buffering: no` (nginx) → tokens arrive in bursts.
- Enabling gzip/HTTP compression → intermediaries buffer until flush thresholds.
- Not sending heartbeats → idle timeouts close long generations.
- Emitting invalid JSON or mixed lines (remember `\n\n` between events).
- Forgetting to cancel GPU work on disconnect → compute leak.

## 15) Integration Notes

- **Serving layer**: expose SSE publicly; internally use gRPC streaming worker shards.
- **Scheduler**: maintain phase-aware queues; emit per-step callbacks for chunk emission.
- **Accounting**: count usage at finalize; optionally stream provisional counters.
- **Speculative decoding**: stream only verified tokens; no client-visible rollbacks.

## 16) Next Steps

- Add **stream-resume**: server keeps a short buffer (e.g., last 2–5 s) keyed by `request_id` to support client reconnection without restarting the decode.
- Integrate **tool-call streaming**: interleave tool-call JSON frames with token deltas under a unified schema.
- Enable **adaptive chunking**: dynamically coalesce deltas under network pressure while preserving TTFT.

**Deliverables**

- A production-ready SSE endpoint with heartbeats, finalization, and robust cancellation.
- Internal gRPC streaming between router and worker with typed deltas.
- Metrics, traces, and logs for TTFT/ITL/TPOT, plus dashboards and alerts.
