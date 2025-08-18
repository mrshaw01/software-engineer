# Serving LLMs with an OpenAI API–style Interface

This document specifies how to expose a production LLM inference service with an interface compatible with OpenAI’s API surface. It focuses on the `/v1/chat/completions` and `/v1/completions` endpoints, streaming semantics, batching-aware request handling, observability, and operational best practices for high-throughput, low-latency serving.

## 1. Goals and Non-Goals

**Goals**

- Provide drop-in compatibility for common OpenAI client SDKs and community tools.
- Support streaming token output with Server-Sent Events (SSE) and an optional WebSocket mode.
- Map request parameters to an optimized backend that implements prefill/decode separation, dynamic batching, and KV cache reuse.
- Offer robust observability, rate limiting, and error semantics suitable for multi-tenant, multi-model deployments.

**Non-Goals**

- Reimplement every OpenAI endpoint (audio, image generation, fine-tuning) from day one.
- Guarantee perfect parity with undocumented behaviors. Where behavior differs, we document it.

## 2. Endpoint Surface

At minimum, implement:

- `GET /v1/models`
  Lists available models and aliases (e.g., `gpt-oss-20b`, `mixtral-8x7b`).

- `POST /v1/chat/completions`
  Chat interface with role-based messages, tool calling, JSON-mode, and streaming.

- `POST /v1/completions`
  Legacy completion interface (single prompt string). Maintain for compatibility.

- `POST /v1/embeddings` (optional)
  Return embeddings for input strings or tokens.

Recommended common headers:

- `Authorization: Bearer <api_key>`
- `Idempotency-Key: <uuid>` for safe retries of POSTs.
- `X-Request-Trace: <uuid>` for cross-service correlation.

## 3. Request and Response Schemas

### 3.1 `/v1/chat/completions` request

```json
{
  "model": "gpt-oss-20b",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Summarize the paper in 3 bullet points." }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 256,
  "stop": ["\n\nUser:"],
  "stream": false,
  "logprobs": false,
  "top_logprobs": 0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "n": 1,
  "seed": 42,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather by city and date",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "date": { "type": "string", "format": "date" }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "Summary bullets",
      "schema": {
        "type": "object",
        "properties": {
          "bullets": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 3,
            "maxItems": 3
          }
        },
        "required": ["bullets"],
        "additionalProperties": false
      },
      "strict": true
    }
  },
  "metadata": { "tenant_id": "acme", "priority": "high" }
}
```

Notes:

- `messages[].role` ∈ {`system`, `user`, `assistant`, `tool`}.
- `tools` (function calling) is supported; the model may emit `tool_calls` that the client/server executes, then append a `tool` role message with the function result and continue generation.
- `response_format` supports `"type": "json_object"` or `"type": "json_schema"`; `"strict": true` enforces well-formed JSON that validates against the schema.

### 3.2 `/v1/chat/completions` response (non-stream)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1739876543,
  "model": "gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "• Point 1\n• Point 2\n• Point 3",
        "tool_calls": []
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 57,
    "completion_tokens": 58,
    "total_tokens": 115
  },
  "system_fingerprint": "router:1.4.2+engine:v2.11.0"
}
```

### 3.3 Streaming with SSE

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`

Each event chunk:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

Tool-call deltas appear as:

```
data: {"choices":[{"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"index":0}]}
```

### 3.4 `/v1/completions` request (legacy)

```json
{
  "model": "gpt-oss-20b",
  "prompt": "Write a haiku about the sea.",
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": false
}
```

### 3.5 Error schema

HTTP codes: `400`, `401`, `403`, `404`, `409`, `422`, `429`, `500`, `503`.

```json
{
  "error": {
    "message": "Rate limit exceeded for tenant=acme.",
    "type": "rate_limit_error",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

Recommended headers on `429`: `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## 4. Backend Architecture

### 4.1 Control Plane

- **Router/Gateway**: Terminates TLS, authenticates, validates requests, normalizes parameters, applies quotas, and forwards to a model-specific queue.
- **Model Registry**: Maps public model names/aliases to engine configurations (weights path, tensor-parallel degree, quantization, scheduling policy).
- **Scheduler/Batcher**: Implements dynamic batching with separate prefill and decode queues. Enforces per-tenant fairness and priorities.
- **Workers**: Run the inference engine (e.g., custom engine, vLLM-like) with Paged/KV cache, FlashAttention kernels, quantized matmuls, CUDA/HIP Graphs.

### 4.2 Data Plane Execution

1. **Prefill**

- Tokenize inputs, build attention mask, run forward pass across all layers, store KV to cache.

2. **Decode**

- Autoregressive loop; each step consumes last token, reuses KV, and produces next token.
- The batcher groups concurrent decode steps across requests to maximize GPU utilization.

3. **Stream**

- Emit SSE chunks per request as tokens are produced; include usage counters once finalized.

### 4.3 Scheduling Considerations

- **Admission Control**: Bound `max_tokens`, context length, and cumulative GPU budget.
- **Priorities**: Use weighted fair queuing per tenant or request `metadata.priority`.
- **Preemption**: Allow low-priority requests to yield decode slots to urgent ones.
- **Cache Policy**: Size KV cache, apply LRU/age-based eviction across sessions.

## 5. Parameter Mapping and Guardrails

- `max_tokens`: hard clamp at router; reject if `prompt_tokens + max_tokens > engine_context_limit`.
- `temperature`/`top_p`/`top_k`: validate ranges; log warnings for degenerate settings.
- `stop`: limit count and cumulative bytes; apply during decode and at postprocessing.
- `logprobs`: if enabled, define `top_logprobs ∈ [1, 20]`. Return token-wise logprobs in non-stream and stream chunks.
- `seed`: deterministic sampling per request when possible; document limits under batching.

## 6. Tool Calling and JSON Mode

- Support `tools` with JSON Schema parameters. The model may produce `tool_calls` requiring function execution. The server may be:

  - **Client-Orchestrated**: Return `tool_calls` to the client; the client executes tools and resubmits with `role:"tool"` message.
  - **Server-Orchestrated**: The server runs a tool sandbox and auto-continues the conversation by appending the tool result.

- `response_format`:

  - `"type":"json_object"`: loosely enforce JSON output.
  - `"type":"json_schema"` + `"strict":true`: enforce schema-conforming JSON; reject on violation with `422`.

## 7. Observability and SLOs

### 7.1 Metrics (Prometheus-style)

- Request: `requests_total`, `request_duration_seconds`, `active_streams`.
- Tokens: `tokens_input_total`, `tokens_output_total`, `tps_decode`, `latency_per_token`.
- Batching: `batch_size_prefill`, `batch_size_decode`, `queue_depth_prefill/decode`.
- Cache: `kv_cache_bytes`, `kv_hits`, `kv_evictions`.
- Errors: `errors_total{code}`, `rate_limit_dropped_total`.
- GPU: `gpu_utilization`, `sm_efficiency`, `mem_bw`, `oom_events`.

### 7.2 Logging

- Structured JSON with `request_id`, `tenant_id`, `model`, `latency_ms`, `tokens_in/out`, `finish_reason`.
- Anonymize or hash user content where required by policy.

### 7.3 Tracing

- OpenTelemetry spans: gateway → router → scheduler → worker.
- Annotate steps: tokenize, prefill, first-token latency (TTFT), decode.

**SLO examples**

- P99 TTFT ≤ 800 ms for ≤ 2k context.
- P99 end-to-end latency ≤ 5 s for ≤ 128 output tokens.

## 8. Multi-Tenancy, Quotas, and Limits

- API keys map to tenants; apply per-tenant QPS and TPM/TPD token budgets.
- Enforce concurrent stream limits per key.
- Soft vs. hard limits:

  - Soft: return `429` with reset headers.
  - Hard: `403` for disallowed models or plan.

## 9. Security and Privacy

- TLS required; reject plaintext except health checks in private networks.
- Key formats: random 32–64 bytes; support key rotation and scopes (model-level, environment).
- CORS: allow-list origins for browser usage.
- Data retention: configurable `store=false` to drop inputs/outputs; redact secrets.
- Audit logs: admin accesses, model changes, quota updates.

## 10. Versioning and Model Aliases

- Stable aliases (e.g., `gpt-oss-20b`) point to the current recommended build.
- Pinned versions (e.g., `gpt-oss-20b-2025-08-01`) for reproducibility.
- Deprecation policy: announce 30+ days in advance; expose `models[].owned_by`, `models[].deprecation_date`.

## 11. Error Handling and Retries

- Retries:

  - Safe with `Idempotency-Key` for transport/timeout errors.
  - Do not retry on `4xx` except `409` (conflict) per documented guidelines.

- Common codes:

  - `400` invalid parameters, schema violations.
  - `401/403` auth issues.
  - `404` model not found.
  - `409` request conflicts (e.g., duplicate idempotency key with mismatched body).
  - `422` JSON schema or tool parameters invalid.
  - `429` rate limit/queue full.
  - `500/503` backend errors or capacity.

## 12. Deployment Topology

- Stateless gateways behind a load balancer.
- Routers as horizontally scaled services with sticky routing per session/tenant (optional).
- Workers pinned to accelerators (GPU/NPU). Configure tensor-parallel, pipeline-parallel, quantization, and memory pools.
- Storage:

  - Model artifacts in object storage with local SSD cache.
  - Optional Redis for rate limiting, idempotency, and session metadata.

- Blue/green or canary rollouts per model build; health checks probe both prefill and decode.

## 13. Minimal Server Examples

### 13.1 FastAPI + SSE (chat completions, stubbed engine)

```python
# app.py
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import time
import uuid

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str | None = None

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 256
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0

def fake_engine_stream(prompt, max_tokens):
    yield {"delta": {"role": "assistant", "content": ""}}
    text = "Hello world from a streaming stub."
    for ch in text[:max_tokens]:
        time.sleep(0.01)
        yield {"delta": {"content": ch}}
    yield {"delta": {}, "finish_reason": "stop"}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Simple input validation
    if len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="messages required")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not req.stream:
        text = "Hello world from a non-streaming stub."
        resp = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": None
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(text.split()), "total_tokens": 10 + len(text.split())}
        }
        return JSONResponse(resp)

    def sse():
        for chunk in fake_engine_stream(req.messages[-1].content or "", req.max_tokens):
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": chunk.get("delta", {}), "finish_reason": chunk.get("finish_reason")}],
            }
            yield f"data: {json.dumps(data, separators=(',',':'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
```

Run:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 13.2 `curl` examples

Non-stream:

```
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-test" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hi"}],"stream":false}'
```

Stream:

```
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-test" -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

## 14. Backend Integration Checklist

- Tokenizer parity with model training (e.g., BPE/SentencePiece, special tokens).
- Prompt formatting for chat templates (system/user/assistant delimiting).
- Stop sequence handling across byte-level and token-level boundaries.
- Logprobs support: ensure decode kernels return logits; compute stable log-softmax.
- KV cache sizing and eviction policy configurable per model.
- Model warmup and graph capture for steady-state latency.

## 15. Rate Limiting and Fairness

- Token-based quotas: `TPM` (tokens per minute) and `RPD` (requests per day).
- Concurrency caps on active streams per key.
- Queue backpressure:

  - `429 queue_full` when prefill queue exceeds threshold.
  - Optional `X-Queue-Position` header for transparency.

## 16. Compatibility Notes and Pitfalls

- `n > 1` with `stream=true`: send interleaved or per-choice chunks. Document chosen behavior (recommend: per-choice chunks with `index`).
- `tool_calls` and streaming: send incremental `arguments` strings; ensure they combine into valid JSON.
- JSON-mode with sampling: if strict JSON is required, consider constrained decoding or temperature clamp near 0.
- `user` field (string) may be sent by clients; treat as opaque metadata and propagate to logs.

## 17. Testing and Conformance

- Golden tests for request/response JSON shape and error codes.
- Fuzz tests for malformed parameters, large prompts, adversarial stop sequences.
- Load tests (prefill-heavy vs. decode-heavy) to measure TTFT, tokens/sec, and P99 latency.
- Compatibility tests with open-source OpenAI SDKs (Python, JS) and LangChain/LiteLLM.

## 18. Configuration Reference (Suggested Defaults)

- `max_request_tokens`: 8192
- `max_output_tokens`: 1024
- `temperature`: 0.7
- `top_p`: 1.0
- `presence_penalty`/`frequency_penalty`: 0
- `timeout_ms`: 60000
- `stream_heartbeat_ms`: 15000 (optional comment frames)
- `kv_cache_budget_per_worker`: size by model and batch target
- `prefill_batch_target`: hardware-dependent (e.g., 8–64 req)
- `decode_batch_target`: hardware-dependent (e.g., 32–256 tokens/step aggregated)

## 19. Rollout and Safety

- Canary by tenant or traffic percentage; monitor TTFT, P99, error rate, and GPU OOM.
- Implement circuit breakers on worker errors and automatic failover to standby pools.
- Maintain compatibility tests and rollback plans per model alias.

## 20. Appendix: `/v1/completions` response (legacy)

```json
{
  "id": "cmpl-xyz789",
  "object": "text_completion",
  "created": 1739876543,
  "model": "gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "text": "Ocean whispers soft,\nTides cradle moonlight and sand—\nStars drink the salt air.",
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 22,
    "total_tokens": 30
  }
}
```

## 21. Appendix: Embeddings (optional)

Request:

```json
{
  "model": "gpt-oss-20b-embed",
  "input": ["hello world", "xin chao"],
  "encoding_format": "float"
}
```

Response:

```json
{
  "object":"list",
  "data":[
    {"object":"embedding","index":0,"embedding":[0.01, -0.02, ...]},
    {"object":"embedding","index":1,"embedding":[0.03, 0.04, ...]}
  ],
  "model":"gpt-oss-20b-embed",
  "usage":{"prompt_tokens":5,"total_tokens":5}
}
```

### Summary

This specification defines a pragmatic, production-ready OpenAI-style interface layered on top of a high-performance inference backend. It balances client compatibility, throughput-oriented scheduling, and strict operational controls to support multi-tenant, large-scale deployments.
