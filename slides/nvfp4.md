---
theme: default
title: Writing GPU Kernel for NVFP4 on Blackwell
fonts:
  provider: none
  sans: source-sans-3-variable
  serif: source-serif-4-variable
  mono: source-code-pro-variable
  local: source-sans-3-variable,source-serif-4-variable,source-code-pro-variable
layout: cover
---

# **Writing GPU Kernels for NVFP4 on Blackwell**
Amandeep Singh

---

## Quantization

- Map high-precision values (FP32/FP16) → low-precision (INT8/INT4/FP4)
- **Why it works**: model weights are approximately **normally distributed** with small variance
  - Most values cluster near zero — few bits needed to represent them
  - Outliers are rare and can be handled with scaling factors
- Goal: find quantized representation $\hat{W}$ that minimizes divergence from original $W$

---

## Why Quantization Works — KL Divergence

- We want the quantized model's output distribution $Q$ to stay close to the original $P$
$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$
- Because weight distributions are **bell-shaped and concentrated**, quantization bins align well with where most of the probability mass lies
- Small $D_{KL}$ ⟹ negligible accuracy loss even at 4-bit precision
- Scaling factors (per-tensor or per-group) absorb outliers, keeping $D_{KL}$ low

---

## Float Zoo

| Format | Bits | Exponent | Mantissa | Range | Use case |
|--------|------|----------|----------|-------|----------|
| FP32   | 32   | 8        | 23       | ±3.4e38 | Training (baseline) |
| BF16   | 16   | 8        | 7        | ±3.4e38 | Training / inference |
| FP16   | 16   | 5        | 10       | ±65504 | Mixed-precision training |
| FP8 (E4M3) | 8 | 4       | 3        | ±448   | Inference |
| FP8 (E5M2) | 8 | 5       | 2        | ±57344 | Gradients |
| **NVFP4 (E2M1)** | **4** | **2** | **1** | **±6** | **Inference (Blackwell)** |

- Trend: trade precision for throughput — 2× ops/bit halved
- NVFP4 needs a **per-group scale factor** (FP8) to compensate for tiny range

---

## Writing...

---
