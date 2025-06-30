# Tenstorrent LLM Inference – Technical Deep Dive

Tenstorrent is an AI hardware startup building next-generation neural processors, with a strong focus on large language model (LLM) inference. The company’s strategy combines **innovative custom AI chips** (“Wormhole” first-generation PCIe cards) and a **fully open-source software stack** to deliver high-performance, scalable LLM inference. Tenstorrent’s hardware (based on proprietary _Tensix_ cores) has been demonstrated running state-of-the-art models like Meta’s LLaMA-70B and Alibaba’s Qwen-72B, both in single-node workstations and multi-board clusters. By 2025, Tenstorrent has open-sourced major components of its inference stack – from low-level kernel libraries to model serving frameworks – aiming to cultivate an ecosystem that can compete with Nvidia’s CUDA in flexibility and performance. This report provides a comprehensive technical deep dive into Tenstorrent’s LLM inference efforts, covering the hardware architecture, core software repositories, kernel and compiler optimizations, supported models, performance benchmarks, and future roadmap.

<div align="center">
    <img src="images/Metalium-vs-TTNN.webp" alt="ttnn + tt-metal" title="ttnn + tt-metal"/>
    <p><em>ttnn + tt-metal</em></p>
</div>
