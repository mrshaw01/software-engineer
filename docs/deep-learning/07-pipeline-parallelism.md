### Pipeline Parallelism in Deep Learning

Pipeline parallelism is a technique to divide a large model across multiple GPUs to enable efficient training when the model is too large to fit into a single GPU's memory. Instead of replicating the entire model on each GPU (as in data parallelism), the model is partitioned into **sub-models**, each assigned to a different GPU. Training proceeds in a pipelined fashion, where GPUs process different parts of the input simultaneously.

### **Key Concepts**

1. **Model Partitioning**

   - The model is split into **multiple stages**, each containing a subset of layers.
   - Each GPU is responsible for forward and backward computations of its assigned stage.

2. **Pipeline Execution**

   - During training, inputs are divided into **micro-batches**.
   - As GPU 0 finishes the forward pass for a micro-batch, it sends the output to GPU 1, which starts its forward pass while GPU 0 begins the next micro-batch.
   - Similarly, backward passes are pipelined in reverse order.

### **Challenges: Pipeline Bubbles**

- **Bubbles** refer to idle GPU time caused by pipeline dependencies, especially at the start and end of each iteration.
- The **backward pass typically takes longer** than the forward pass, increasing the idle time for some GPUs.

### **Micro-Batching to Reduce Bubbles**

- **Micro-batches** divide a mini-batch into smaller chunks.
- By processing more micro-batches simultaneously in different pipeline stages, GPU idle time (bubbles) is reduced.
- However, **too small micro-batches increase overhead** and reduce intra-batch parallelism.

### **Trade-offs**

- **Smaller micro-batches**: Reduce bubbles but increase overhead and reduce resource utilization.
- **Larger micro-batches**: Improve throughput but lead to more pipeline bubbles.
- Selecting an optimal micro-batch size depends on the **system architecture and model characteristics**.

### **Advanced Optimizations**

- Techniques like **PipeDream** improve pipeline efficiency further by optimizing **partitioning and scheduling**, overlapping forward and backward computations more effectively.

### **Benefits of Pipeline Parallelism**

- Enables training of very large models beyond single GPU memory.
- Increases overall hardware utilization when tuned properly.
- Can be combined with **data parallelism** and **tensor parallelism** for large-scale distributed training.
