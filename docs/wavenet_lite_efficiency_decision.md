# WaveNet Lite CH12 — Efficiency Decision (F-A7 Resolution)

**Date:** 2026-07-20
**Status:** RESOLVED — Desfecho B (Revert T6.S3.2)
**Reference:** EPIC-7, Sprint S2, T7.S2.3

## Executive Summary

T6.S3.2 (pad-to-16 structural) was a 537-line change across 19 files that added stride-16
internal buffers, 3 new kernel modules (`stride16.rs`), and conditional `if CH == 12` dispatch
to eliminate cache-line splits in WaveNet Lite CH12. After dedicated profiling and structural
ASM analysis (T7.S2.1 + T7.S2.2), the pad-to-16 showed **zero measured performance benefit**
(52.2 µs → 52.3 µs, within noise).

The decision is to **revert T6.S3.2** (Desfecho B), preserving only the weight padding from the
original EPIC-2 (T2.S3.1-T2.S3.3) which delivered the −19.7% gain and is not affected by this
finding.

## Background

### What T6.S3.2 Changed

| Component                     | Change                                | Purpose                                  |
| ----------------------------- | ------------------------------------- | ---------------------------------------- |
| `effective_stride` helper     | Added to `common.rs`                  | Returns 16 for CH=12, CH otherwise       |
| `gemm_batch/stride16.rs`      | New module (152 lines)                | Padded GEMM for stride-16 12×12          |
| `gemv/stride16.rs`            | New module (140 lines)                | Padded GEMV/broadcast for stride-16      |
| `layer.rs`                    | Conditional dispatch `if CH == 12`    | Routes to stride16 or generic kernels    |
| `layer_array.rs`              | Conditional dispatch `if IN/CH == 12` | Rechannel dispatch to stride16           |
| `conv1d.rs`, `conv1d_dual.rs` | Added `stride` parameter              | Passes effective stride through          |
| `standard.rs` (dispatcher)    | Buffer allocation with `stride`       | Allocate 16× buffers for CH=12 layers    |
| Test tolerance                | Lite parity relaxed to 1e-6           | Due to FMA order differences in stride16 |

### What EPIC-2 Original (T2.S3.1-T2.S3.3) Changed (PRESERVED)

The original EPIC-2 padded convolution *weights* from stride 12 to 16 in the model loader.
This eliminated load misalignment in the Conv1D hot-path (weights accessed per-tap, per-channel).
This change is NOT reverted — it delivered −19.7% latency reduction for Lite (68.5 → 52.2 µs)
and remains bit-exact.

## Investigation Results (T7.S2.2)

The full analysis is documented in `docs/wavenet_lite_ch12_profiling.md`.

### Hypothesis Evaluation

| #   | Hypothesis                                             | Verdict                                                                                     |
| --- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| 1   | Fixed overhead per layer dominates with fewer channels | **SUSTAINED** — Overhead/Useful ratio 2.40 (Lite) vs 1.07 (Standard)                        |
| 2   | SIMD domain transition (AVX↔SSE) in stride16 kernels   | **PARTIALLY SUSTAINED** — All VEX-encoded; no legacy penalty. Cost is lane underutilization |
| 3   | Original ≤42 µs target was incorrect                   | **SUSTAINED** — Not achievable for CH=12 in AVX2 without architectural redesign             |

### Key Data Points

| Metric                 | Standard CH16 | Lite CH12 | Ratio |
| ---------------------- | ------------- | --------- | ----- |
| Instructions per layer | 1,815         | 3,974     | 2.19× |
| Overhead %             | 34.5%         | 54.0%     | 1.56× |
| Overhead/Useful ratio  | 1.07          | 2.40      | 2.24× |
| XMM/YMM ratio          | 5.7%          | 23.1%     | 4×    |
| vzeroupper calls       | 30            | 79        | 2.6×  |
| Stack frame            | 872B          | 1,448B    | 1.66× |
| Latency                | 37.6 µs       | 52.3 µs   | 1.39× |
| µs/MMAC                | 2,449         | 6,057     | 2.47× |

### Why Pad-to-16 Didn't Help

The profiling (T7.S2.1) showed `fused_gemm_residual_batch_f32_12x12_padded` (the T6.S3.2
artifact) consumed only 5.11% of the Lite workload. The dominant bottleneck is NOT cache-line
splits in buffer loads/stores — it's the structural overhead of CH=12 not fitting evenly into
AVX's 8-wide SIMD lanes:

1. 12 channels require 8+4 split (YMM+XMM) — wasting 4 YMM lanes per vector operation
2. Tap gathering needs blend/perm instructions (`vblendps`, `vmaskmovps`, `vshufps`) — 3% overhead
3. Larger stack frame (1,448B vs 872B) — more register spilling
4. Fixed per-layer overhead (prologue, bounds-checks, dispatch) amortized over fewer MAC/layer

None of these factors are addressed by padding buffer strides to 16.

## Decision: Desfecho B — Revert T6.S3.2

### Rationale

1. **No benefit:** T6.S3.2 added 537 lines of complexity with zero measurable performance gain
2. **Engineering cleanliness:** Code that doesn't pay its complexity should be removed
3. **Tolerance restoration:** Lite parity test returns to 1e-7 (same as all other SKUs)
4. **No regression:** Lite at 52.7 µs post-revert vs 52.2 µs pre-T6.S3.2 (change within noise)

### What Was Reverted

- Removed `src/math/gemm/gemm_batch/stride16.rs` and `src/math/gemm/gemv/stride16.rs`
- Removed `effective_stride` function from `src/models/wavenet/common.rs`
- Reverted all `if CH == 12` / `if IN == 12` dispatch branches in `layer.rs` and `layer_array.rs`
- Reverted `stride` parameter passing through `conv1d.rs`, `conv1d_dual.rs`
- Reverted buffer allocations from stride-16 to CH in dispatcher, model, and tests
- Removed T6.S3.3 padding-lanes proptest (no longer applicable)
- Restored Lite dynamic parity tolerance to 1e-7

### What Was Preserved

- EPIC-2 weight padding (T2.S3.1-T2.S3.3) — the 16-wide interleaved weight layout in `standard.rs`
- All goldens, quality contracts, ESR baselines — unchanged

## Gate Results

| Gate                               | Result                             |
| ---------------------------------- | ---------------------------------- |
| `utils/lints.sh`                   | ✅ Passed                          |
| `utils/tests-quick.sh`             | ✅ All 18 tests passed             |
| `utils/quality-dashboard.sh`       | ✅ All metrics within tolerance    |
| `cargo test dynamic_parity`        | ✅ All SKUs pass with 1e-7         |
| `cargo test thp_coherence`         | ✅ Passed                          |
| `cargo test rt_deadline`           | ✅ Passed                          |
| `cargo bench RT_WaveNet_Lite_CH12` | ✅ 52.685 µs (change within noise) |
| `cargo bench RT_WaveNet_Std_CH16`  | ✅ 38.647 µs (no change)           |

## Conclusion

The Lite CH12 remains at ~52.7 µs — the same performance as post-EPIC-2 (52.2 µs) and
post-T6.S3.2 (52.3 µs), confirming that the pad-to-16 structural changes were a no-op for
performance. The ≤42 µs meta is formally abandoned as not achievable without architectural
redesign (AVX-512 with mask registers, or topology changes to multiples of 8 channels).

The Lite SKU operates at 96.1% headroom from the 1333 µs RT deadline — well within safe
operating margins. No functional urgency exists for further optimization.

---

*Decision documented per the same standard as `docs/pw_dual_stream_architecture.md`.*
*See `docs/wavenet_lite_ch12_profiling.md` for the full technical ASM analysis supporting this decision.*
