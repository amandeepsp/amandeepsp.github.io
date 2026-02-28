---
title: "Twelve Attempts at an FP4 Kernel"
subTitle: "A worklog of NVFP4 kernels, failed experiments, and one stubborn memory bus"
publishDate: "Feb 28 2026"
tags: [ml, cuda, gpu]
toc: true
featured: false
seo:
  description: "Writing an NVFP4 GEMV kernel for Blackwell B200. Twelve attempts, five failed optimizations, and lessons from a memory-bound wall."
---

I recently participated in [GPU Mode's](https://discord.gg/gpumode) NVFP4 Kernel Hackathon. The goal was to write CUDA kernels for Blackwell's new 4-bit floating point format (NVFP4) and get as close as possible to the hardware's speed of light. The competition ran on B200 GPUs, and there were four tasks in total: matrix-vector multiplication (GEMV), matrix-matrix multiplication (GEMM), gated GEMM (with SiLU activation), and grouped GEMM. All tasks use [block-scaled FP4](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout) inputs where every 16 elements of `e2m1` data share a single `e4m3` FP8 scale factor, with `fp16` output. I was able to complete the GEMV and GEMM tasks but ran out of time for the gated and grouped variants.

If you want background on CuTe's layout algebra (which underpins a lot of what follows), I wrote about it [here](/blog/layout-algebra).

## The Task

For GEMV we were given a matrix $A$ of shape $M \times K \times L$ and a vector $B$ of shape $1 \times K \times L$ (both in packed FP4 with FP8 scale factors), compute $C = A \cdot B^T$ of shape $M \times 1 \times L$ in FP16. The ranking metric was the geometric mean of benchmark times, measured against a speed-of-light analysis based on `max(FFMA math throughput, DRAM memory throughput)` of the B200 at 1.5GHz clock:

| M | K | L | Speed of Light (µs) |
|-----|-------|---|-----|
| 7168 | 16384 | 1 | 8.6 |
| 4096 | 7168 | 8 | 17.3 |
| 7168 | 2048 | 4 | 4.3 |

## Optimization Journey

This is where I spent most of my time. Twelve attempts, of which only one really worked well.

### CuTe Python DSL (Attempts 1-4)

My first instinct was to use CUTLASS's Python DSL since I had been experimenting with CuTe's layout system already. The initial approach was straightforward. One thread per output row element, iterating over K, decoding FP4 to FP16, multiplying by scale factors, and accumulating.

```python
@cute.kernel
def _kernel(self, a, b, sfa, sfb, c):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # ... tensor setup, local_tile ...

    global_m = bidx * self.b_m + tidx
    if global_m < M:
        tCrC = cute.zeros_like(tCgC, cutlass.Float32)

        for k in range(k_tiles):
            # Load FP4 values, decode to FP16
            a_val = tAgA[tidx, None, k].load().to(cutlass.Float16)
            b_val = tBgB[0, None, k].load().to(cutlass.Float16)
            sfa_val = tAgSFA[tidx, None, k].load().to(cutlass.Float32)
            sfb_val = tBgSFB[0, None, k].load().to(cutlass.Float32)

            for i in cutlass.range_constexpr(self.b_k):
                tCrC += (a_val[i] * b_val[i]) * (sfa_val[i] * sfb_val[i])

        tCgC.store(tCrC.to(cute.Float16))
```

This produced correct results but the performance was not competitive. Each thread walks the entire K dimension sequentially with no parallelism along K. Over the next few attempts I experimented with K-dimension tiling and thread configuration but stayed within the same basic structure.

### Split-K with Atomics in CuTe (Attempt 5)

To parallelize the K reduction I split the K dimension across threads and used `atomicAdd` to accumulate partial sums. This required defining custom `@dsl_user_op` functions for operations like `atomic_add_fp32` that CuTe doesn't natively expose:

```python
@dsl_user_op
def atomic_add_fp32(a, gmem_ptr, *, loc=None, ip=None):
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )
```

The atomics were too expensive. The next attempt (attempt 6) replaced them with warp-shuffle reductions. 128 threads split into 4 warps, each warp handling one M row with 32-lane K tile splitting. This was better, but I was hitting a wall on the compute side.

The inner loop needs packed FP16 fused multiply-add (PTX: `fma.rn.f16x2`) to decode FP4 pairs, scale them, and accumulate efficiently. The CuTe Python DSL exposes `fma_packed_f32x2` but has no equivalent `fma_packed_f16x2` [wrapper](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py). Scalar FP16 arithmetic works through MLIR's `arith` dialect (`arith.mulf`, `arith.addf`), but whether the backend fuses `a * b + c` into an actual `fma.rn.f16x2` is up to LLVM optimization passes, not something you can control or rely on. I tried writing `llvm.inline_asm` wrappers for the `half2` operations but couldn't get them to work, and moved on to C++.

In hindsight this was a skill gap, not a DSL limitation. Another competitor placed in the top 10 (21.6µs) using a pure CuTe kernel that emits the entire decode-FMA-reduce pipeline as a single `llvm.inline_asm` block with `cvt.rn.f16x2.e2m1x2` and `fma.rn.f16x2`, exactly what I was trying to do. The inlining is straightforward once you get the constraint strings right. I just didn't push through it at the time.

### The Switch to Raw CUDA (Attempt 7)

I scrapped the CuTe approach and rewrote from scratch in raw CUDA C++, loaded via `torch.utils.cpp_extension.load_inline`.

The core idea is to assign 32 threads (one warp) per output row. Each thread handles a strided slice of the K dimension, accumulates locally, then the warp reduces via `__shfl_down_sync`. I used 128 threads per block with 4 rows per block.

For FP4 decoding I used Blackwell's new intrinsics. Each byte packs two FP4 values, and `__nv_cvt_fp4x2_to_halfraw2` converts them to a `half2` pair in a single instruction[^1]:

```cpp
__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
}
```

The key function is the packed `half2` dot product over a 4-byte (8 FP4 element) chunk. This is where the real speedup came from. The rest of the kernel structure (strided K loop, warp reduction) is fairly standard, but doing the decode-scale-multiply pipeline entirely in paired `half2` operations avoids the scalar FP32 overhead of the CuTe attempts:

```cpp
__device__ __forceinline__ __half2 dot_scaled_4bytes(
    uchar4 a4, uchar4 b4, __half2 scale_h2
) {
    __half2 acc0 = __hmul2(decode_fp4x2(a4.x),
                           __hmul2(decode_fp4x2(b4.x), scale_h2));
    __half2 acc1 = __hmul2(decode_fp4x2(a4.y),
                           __hmul2(decode_fp4x2(b4.y), scale_h2));
    acc0 = __hfma2(decode_fp4x2(a4.z),
                   __hmul2(decode_fp4x2(b4.z), scale_h2), acc0);
    acc1 = __hfma2(decode_fp4x2(a4.w),
                   __hmul2(decode_fp4x2(b4.w), scale_h2), acc1);
    return __hadd2(acc0, acc1);
}
```

Each scale factor group covers 16 FP4 elements = 8 bytes. The outer loop loads two `uchar4` reads per group (4 bytes each, 8 FP4 values per load), calls `dot_scaled_4bytes` on each, and accumulates into FP32:

```cpp
#pragma unroll 4
for (int sf = tid; sf < K_sf; sf += THREADS_PER_ROW) {
    float scale = decode_fp8(__ldg(&row_sfa[sf])) *
                  decode_fp8(__ldg(&batch_sfb[sf]));
    __half2 scale_h2 = __halves2half2(__float2half(scale),
                                       __float2half(scale));

    int byte_base = sf << 3;
    uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a[byte_base]);
    uchar4 b4_0 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base]);
    uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a[byte_base + 4]);
    uchar4 b4_1 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base + 4]);

    __half2 r0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2);
    __half2 r1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2);
    __half2 sum = __hadd2(r0, r1);
    float2 f = __half22float2(sum);
    acc += f.x + f.y;
}
```

After accumulation, the warp reduces with shuffles:

```cpp
row_sum += __shfl_down_sync(0xffffffff, row_sum, 16);
row_sum += __shfl_down_sync(0xffffffff, row_sum, 8);
row_sum += __shfl_down_sync(0xffffffff, row_sum, 4);
row_sum += __shfl_down_sync(0xffffffff, row_sum, 2);
row_sum += __shfl_down_sync(0xffffffff, row_sum, 1);

if (lane == 0) {
    c[c_idx] = __float2half(row_sum);
}
```

Results:

| M | K | L | My kernel (µs) | Speed of Light (µs) | Ratio |
|-----|-------|---|------|------|-------|
| 7168 | 16384 | 1 | 26.7 | 8.6 | 3.1x |
| 4096 | 7168 | 8 | 45.1 | 17.3 | 2.6x |
| 7168 | 2048 | 4 | 16.4 | 4.3 | 3.8x |

Roughly 3x off speed of light. This became my baseline.

### The Failed Experiments

Over the next five attempts, I tried every optimization I could think of. All of them made things worse or had zero effect.

In attempt 8, I went back to split-K with atomics, this time in C++. I tiled the K dimension across blocks and used `atomicAdd` to accumulate partial sums into an FP32 intermediate buffer with a final FP32-to-FP16 conversion pass. The atomic contention and extra memory traffic outweighed any parallelism benefit. The kernel is memory-bound, and adding more blocks just means more scheduling overhead and more atomic serialization at the same memory addresses.

Attempt 9 tried wider vectorized loads. Instead of two `uchar4` loads (32 bits each), I used a single `uint2` load (64 bits). This was 16-25% slower. Extracting individual bytes from a `uint2` requires bitwise operations (`& 0xFF`, `>> 8`, etc.) and `make_uchar4()` calls. The compiler already optimizes two consecutive `uchar4` loads into efficient memory transactions. I was adding instruction overhead to "save" an instruction the compiler was already handling.

Attempt 10 was the worst regression at +32-55%. I tried four independent accumulator chains to hide FMA latency through instruction-level parallelism. But the kernel is *memory-bound*, not compute-bound. FMA latency hiding is irrelevant when every cycle is waiting on memory. The four accumulator chains increased register pressure enough to cause spilling, and the strided K access pattern (`THREADS_PER_ROW * 4` instead of `THREADS_PER_ROW`) reduced memory coalescing. I was optimizing for the wrong bottleneck.

In attempt 11 I tried tuning register count and block size. Reducing `-maxrregcount` from 80 to 64 had zero effect since the kernel naturally uses fewer than 64 registers. `BLOCK_SIZE=256` with `ROWS_PER_BLOCK=8` also changed nothing, because the 32-threads-per-row warp structure is what matters, not the block size. Bumping to `#pragma unroll 8` instead of `unroll 4` dropped performance by 5-87% due to register pressure and I-cache misses.

Finally, attempt 12 tried software pipelining with an explicit prologue to prefetch the next K tile into registers while computing the current one. On the B200's memory subsystem, the hardware prefetcher combined with `__ldg` cache hints is already doing this. Manual pipelining just doubled register pressure for data that was already on its way.

### Why 3x Off?

The kernel is fundamentally memory-bound. FP4 data is tiny (4 bits per element), and even with block scaling overhead, the arithmetic intensity is low for GEMV. The speed-of-light analysis is based on DRAM bandwidth limits. Even the top solutions only reached ~2x off speed of light. The gap from 2x to 1x is genuinely hard for GEMV and may require approaches beyond what any competitor used.

The gap between my 3x and the winners' 2x came down to execution details I cover in the next section: cache policies, load widths, register pressure, and compile-time specialization. I didn't invest enough time into understanding the Nsight Compute profile for my kernel. Had I done that after attempt 7, it would have told me immediately that the kernel was memory-bound and that compute-side optimizations (ILP, wider instructions) were pointless. I instead learned this the hard way through attempts 8-12.

## What the Top Solutions Did Differently

After the hackathon ended I studied the top 3 solutions (all clustered around 18.5µs geometric mean, roughly 2x speed of light vs my 3x). They shared several techniques I hadn't used.

All three wrote their load and decode paths in raw PTX rather than using C intrinsics. Where I used `__nv_cvt_fp4x2_to_halfraw2`, they wrote `cvt.rn.f16x2.e2m1x2` directly. Where I used `__ldg`, they wrote `ld.global` with explicit qualifiers. This gives precise control over instruction selection and scheduling that the C intrinsics abstract away.

The biggest gap was cache policy control. All top solutions used different cache hints for A (the matrix, streamed once) vs B (the vector, reused across all rows). For A they used `L1::no_allocate` to avoid polluting L1 with data that won't be reused. For B they used `L1::evict_last` to keep it hot in cache since every row reads the same vector. My `__ldg` just requests a generic read-only cache path with no distinction between streaming and reusable data.

They also used much wider vectorized loads. I was loading with `uchar4` (32-bit). The top solutions loaded with `ld.global.v2.u64` (128-bit) and even `ld.global.v4.u64` (256-bit), fetching 32 or 64 FP4 values in a single memory transaction. This is the kind of wider load that actually works. Unlike my attempt 9 with `uint2`, the top solutions decoded using PTX byte unpacking (`mov.b32 {tmp0, tmp1, tmp2, tmp3}, %reg`) which avoids the bitwise extraction overhead that killed my approach.

Rather than one generic kernel, the top solutions templated on the exact K dimension and dispatched at launch time. This lets the compiler fully unroll the K loop with a known trip count and select optimal register allocation per problem size. The rank 1 solution went further with per-K cache hint tuning, using different `ld.global` qualifiers for K=3584, K=8192, and K=1024.

Register budgets were also much tighter. The rank 1 solution used `-maxrregcount=32`, rank 3 used 45. I used 80. Lower register counts increase occupancy (more warps in flight), which is what a memory-bound kernel needs to hide memory latency. My attempt 11 tested this but only went down to 64, not aggressive enough.

The rank 2 solution also processed `BLOCK_M` rows per thread block where threads reading B data are shared across rows. This amortizes the B vector load cost. My kernel loaded B independently per warp, wasting bandwidth on redundant reads of the same vector.

One thing that surprised me; a pure PyTorch solution using `torch._scaled_mm` with multi-stream parallelism across the L dimension scored 22.4µs. No custom kernels at all, just calling into cuBLAS's FP4 path with the right scale factor layout. That's within 20% of the top 3 PTX solutions and faster than my hand-written C++ kernel.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PTX is all you need</p>&mdash; GPU MODE (@GPU_MODE) <a href="https://twitter.com/GPU_MODE/status/2025235208006762703?ref_src=twsrc%5Etfw">February 21, 2026</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## Takeaways

The single most important thing I could have done after attempt 7 was run Nsight Compute and confirm the kernel was memory-bound. That would have saved me from attempts 8-12. Instead I optimized based on intuition, and intuition was wrong. Split-K doesn't help memory-bound kernels. Wider loads only help when data can be used directly without unpacking. ILP is irrelevant when the bottleneck is memory. Register tuning does nothing if you're already under the limit. Software pipelining is redundant when hardware prefetch is sufficient.

The full source code for all attempts is available on [GitHub](https://github.com/amandeepsp/cuda).

[^1]: At the PTX level this maps to `cvt.rn.f16x2.e2m1x2`, which is a single instruction. However, the C intrinsic doesn't always compile down to it cleanly. The rank 2 hackathon winner noted that `__nv_cvt_fp4x2_to_halfraw2()` didn't produce the desired PTX, which is one reason all top solutions wrote inline PTX directly.
