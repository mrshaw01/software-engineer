# LLM Serving Architectures

Efficient serving architectures are critical for deploying large language models in production. These architectures determine how requests are routed, how resources are allocated, and how inference workloads are scheduled. The design must balance latency, throughput, scalability, and fault tolerance while managing the high compute and memory demands of LLMs.

This section outlines common architectural patterns for LLM serving.

## 1. Monolithic Server

A single process manages the entire inference pipeline: request parsing, preprocessing, model execution, and response generation.

**Characteristics:**

- Simple to implement and deploy.
- Direct access to GPU/TPU/NPU memory with minimal communication overhead.
- Limited scalability — cannot easily handle heterogeneous workloads or multiple concurrent models.
- Typically used for research prototypes, small models, or single-tenant deployments.

**Tradeoffs:**
Low latency but poor elasticity. Scaling requires replicating the entire server.

## 2. Worker–Pipeline Model

The model execution is broken into stages (e.g., tokenization, embedding lookup, transformer blocks, sampling). Each stage is handled by specialized workers that form a pipeline.

**Characteristics:**

- Improves hardware utilization by overlapping computation across requests.
- Can support pipelined parallelism (layer partitioning across devices).
- Complexity in scheduling and ensuring balanced workloads.
- Useful when a single model is too large to fit on one device.

**Tradeoffs:**
High throughput for large models, but increased inter-process communication and synchronization overhead.

## 3. Router–Dispatcher Pattern

A lightweight router node receives client requests and dispatches them to backend workers for execution.

**Characteristics:**

- Decouples request handling from model execution.
- Enables multi-model or multi-version serving, with the router applying policies (e.g., A/B testing, load balancing).
- Allows scaling workers independently based on demand.

**Tradeoffs:**
Router introduces a network hop, but provides flexibility and elasticity.

## 4. Microservice Architecture

Inference is decomposed into independent microservices (e.g., tokenization, embedding cache, model execution, post-processing). Each service runs independently and communicates via RPC or message queues.

**Characteristics:**

- Clear isolation between components.
- Easier to update or replace individual services.
- Supports heterogeneous infrastructure (e.g., CPU-based tokenization + GPU inference).
- Potential for increased latency due to inter-service calls.

**Tradeoffs:**
Operational overhead is high, but supports complex enterprise deployments.

## 5. Hybrid Architectures

Many production systems combine elements of the above approaches. Examples include:

- **Router + Monolithic Workers**: A router handles request distribution, while each worker runs a monolithic inference server.
- **Pipeline + Router**: A router balances load across multiple pipelined inference backends.
- **Microservices + Router**: Microservices handle supporting tasks (e.g., caching, logging) while inference workers remain monolithic.

## 6. Multi-Tenant Serving

When serving multiple users or models on shared hardware, the architecture must support isolation, quota enforcement, and fairness.

**Approaches:**

- **Static Partitioning**: Reserve resources per model/user.
- **Dynamic Scheduling**: Share GPUs across tenants with admission control and preemption.
- **Priority Queues**: Assign priority weights to users or request types.

## Summary

- **Monolithic servers** provide simplicity and low latency but scale poorly.
- **Worker–pipeline models** maximize throughput on large models but add scheduling complexity.
- **Router–dispatcher patterns** enable flexible load balancing and multi-model serving.
- **Microservices** improve modularity and maintainability at the cost of higher latency.
- **Hybrid architectures** are common in practice, blending tradeoffs for specific use cases.
- **Multi-tenant serving** adds another layer of complexity, requiring scheduling, quotas, and fairness policies.

The choice of architecture depends on deployment scale, latency constraints, and operational complexity.
