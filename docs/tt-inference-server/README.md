# TT-Inference-Server: Tenstorrent’s LLM Inference Server Overview

- https://github.com/tenstorrent/tt-inference-server

## Overall Purpose

**TT-Inference-Server** is Tenstorrent’s open-source inference server, designed to deploy Large Language Models (LLMs) (and some vision models) on Tenstorrent AI hardware. In essence, it provides a ready-to-run serving stack that integrates Tenstorrent’s custom accelerator libraries with high-level model APIs. The repository contains implementations of various model inference “APIs” (model handlers) which are optimized for Tenstorrent devices, so users can serve models on Tenstorrent’s **Wormhole** accelerators (e.g. **TT-LoudBox**, **TT-QuietBox** desktops or **n150** PCIe cards) with minimal setup. By abstracting the hardware details behind a standard interface (often an OpenAI-compatible REST API), TT-Inference-Server lets developers interact with Tenstorrent-powered models just like they would with a typical GPU-based server, but harnessing Tenstorrent’s efficiency.
