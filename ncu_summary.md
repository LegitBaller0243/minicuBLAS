# Nsight Compute SharedTilingKernel Summary 

## Core Performance Story
1. Primary bottleneck is **underfilled GPU / launch configuration**, not memory bandwidth.
2. Main limiter is **scheduler starvation** (too few eligible warps).
3. Compute pipelines are **underutilized**.
4. Memory is **not saturated** and does not appear to be the dominant bottleneck.
5. Control flow is clean (no divergence issues).

## Key Metrics Table

| Area | Metric | Value | Why It Matters |
|---|---|---:|---|
| Launch / Scale | Grid Size | 256 blocks | Too small for this GPU size. |
| Launch / Scale | Block Size | 256 threads | Reasonable, but total grid is still small. |
| Launch / Scale | SM Count | 132 SMs | Large device; needs more total work. |
| Launch / Scale | Waves Per SM | 0.24 | Very low residency over time. |
| Launch / Scale | Full Waves Across GPU | 0.2 | Explicit Nsight warning: kernel is underfilled. |
| Occupancy | Achieved Occupancy | 24.02% | Far below potential occupancy. |
| Occupancy | Achieved Active Warps / SM | 15.37 | Lower than max 64 warps/SM. |
| Scheduler | No Eligible | 75.41% | Most cycles have no warp ready to issue. |
| Scheduler | Eligible Warps / Scheduler | 0.41 | Very low instruction-ready warp pool. |
| Scheduler | Issued Warp / Scheduler | 0.25 | About one issued instruction every 4.1 cycles. |
| Compute Utilization | SM Busy | 22.12% | Compute hardware is mostly idle. |
| Compute Utilization | Issue Slots Busy | 20.29% | Scheduler issue slots underused. |
| Compute Utilization | Executed IPC (Active) | 0.96 inst/cycle | Limited throughput per active cycle. |
| Roofline | FP32 Peak Achieved | 4% | Low arithmetic throughput vs device peak. |
| Roofline | FP64 Peak Achieved | 0% | No meaningful FP64 utilization. |
| Memory | Memory Throughput | 36.13 GB/s | Moderate, not close to peak pressure. |
| Memory | DRAM Throughput | 0.75% of peak | Strong sign DRAM is not bottleneck. |
| Memory | L2 Hit Rate | 86.38% | Good cache reuse at L2. |
| Memory | L1/TEX Hit Rate | 2.84% | Low L1 hit, but not dominating runtime. |
| Memory | Local Spill Requests | 0 | No local-memory spill penalty. |
| Control Flow | Branch Efficiency | 100% | Branching is efficient. |
| Control Flow | Avg. Divergent Branches | 0 | No warp divergence concern. |

## Nsight Diagnostics to Mention
- **Grid too small warning**: only 0.2 full waves across SMs.
- **Compute under-utilized**: Nsight local speedup estimate ~93.59%.
- **Scheduler bottleneck**: Nsight local speedup estimate ~57.02%.
- **PM sampling warning**: sampling interval too large vs 14.62 us kernel duration.

## Interview Sound Bite
This kernel is primarily **parallelism/latency-hiding limited** (launch + eligibility), not memory-bandwidth limited. The next steps are to increase total concurrent work, raise eligible warps per scheduler, and then re-profile with a smaller PM sampling interval.

---

# RegTiling+VecFlow 2 Summary (Nsight Compute)

## Core Performance Story
1. This kernel is **substantially stronger** than Kernel 1 in raw utilization (compute + scheduler).
2. Main constraints are **very low occupancy** (resource-limited) and some **launch underfill**.
3. Most meaningful optimization signal is **uncoalesced shared-memory access**.
4. Memory bandwidth is active but still not DRAM-saturated.

## Key Metrics Table

| Area | Metric | Value | Why It Matters |
|---|---|---:|---|
| Topline | Duration | 75.87 us | Longer and heavier kernel than Kernel 1. |
| Topline | Compute (SM) Throughput | 70.28% | Compute units are relatively well utilized. |
| Topline | Memory Throughput | 61.11% | Memory subsystem is active. |
| Topline | DRAM Throughput | 2.32% of peak | DRAM is still far from saturation. |
| Compute | SM Busy | 70.28% | Much higher than Kernel 1. |
| Compute | Executed IPC (Active) | 2.97 inst/cycle | Strong instruction throughput. |
| Compute | Dominant Pipe | FMA 57.1% | Math-heavy kernel behavior. |
| Scheduler | One or More Eligible | 74.65% | Schedulers often have work ready. |
| Scheduler | No Eligible | 25.35% | Some stall remains, but far improved. |
| Scheduler | Issued Warp / Scheduler | 0.75 | Good issue rate (vs 0.25 in Kernel 1). |
| Occupancy | Theoretical Occupancy | 12.50% | Hard cap is low before runtime effects. |
| Occupancy | Achieved Occupancy | 6.25% | Very low runtime occupancy. |
| Occupancy | Achieved Active Warps / SM | 4.00 | Weak latency hiding headroom. |
| Occupancy Limiters | Registers / Thread | 224 | Very high register pressure. |
| Occupancy Limiters | Dynamic Shared Mem / Block | 18.82 KB | Shared memory also constrains blocks/SM. |
| Launch | Grid Size | 128 blocks | Fewer blocks than 132 SMs (minor underfill). |
| Launch | Block Size | 128 threads | Fine by itself; total launch count is limiting. |
| Launch | Waves Per SM | 0.48 | Still low total waves. |
| Memory | Memory Throughput | 111.49 GB/s | Solid memory traffic level. |
| Memory | L2 Hit Rate | 95.99% | Very high L2 locality. |
| Memory | L1/TEX Hit Rate | 0% | Likely bypass/pattern-related; not the main issue alone. |
| Memory | Local Spill Requests | 0 | No spill overhead. |
| Control Flow | Avg. Active Threads / Warp | 32 | Full warp utilization. |
| Control Flow | Avg. Not Predicated Off Threads / Warp | 31.49 | Minimal predication loss. |

## Nsight Diagnostics to Mention
- Grid has **128 blocks < 132 SMs** (minor launch underutilization, est. speedup ~3.03%).
- Occupancy warning indicates large potential from raising resident warps (est. local speedup ~87.5%).
- Shared-memory access pattern warning: **uncoalesced shared accesses** causing **1,572,864 excessive wavefronts** (~18% of total), with est. speedup ~17.02%.
- L2 compression currently gives no benefit (small projected gain ~2.19%).

## Reflection
RegTiling+VecFlow 2 is a **higher-throughput, math-heavy kernel** that still leaves performance on the table due to **resource-limited occupancy** (high registers + shared memory) and **shared-memory access inefficiency**. Priority should be to improve shared-memory coalescing first, then reduce per-thread/block resource usage to lift occupancy.

---

# Prioritized Next Fixes

## 1) Increase launch-scale parallelism first (both kernels)
- Raise total concurrent CTAs so the GPU is not underfilled (`grid size`, `waves per SM` too low in both summaries).
- In benchmarking, prioritize larger problem sizes and/or larger batch so block count is comfortably above SM count.
- Re-check: `Waves Per SM`, `No Eligible`, total kernel duration.

## 2) Lift occupancy by reducing per-CTA resource usage (especially RegTiling+VecFlow)
- Reduce register pressure (currently very high) by tuning microtile size and unroll depth.
- Reduce shared-memory footprint per CTA where possible.
- Re-check: `Achieved Occupancy`, `Active Warps/SM`, `No Eligible`.

## 3) Retune tile/microtile shapes for residency vs reuse tradeoff
- Test smaller per-thread microtiles and/or smaller CTA tiles to increase resident CTAs/SM.
- Keep arithmetic intensity high enough, but prefer better scheduler eligibility when occupancy is severely low.
- Re-check: runtime first, then `Issued Warp/Scheduler` and `SM Busy`.

## 4) Fix shared-memory access inefficiency in RegTiling+VecFlow
- Address Nsight warning on uncoalesced shared-memory accesses (excessive wavefronts).
- Validate shared-memory indexing/layout choices (bank-conflict-aware access pattern).
- Re-check: shared-memory diagnostics and kernel duration.

## 5) Keep ILP optimizations constrained by register growth
- Additional unrolling/ILP is useful only if it does not further collapse occupancy.
- Treat register count as a hard budget while tuning.
- Re-check: `Registers/Thread` vs achieved occupancy and total time.

## 6) Use a repeatable profiling loop per change
- For each kernel edit, compare:
  - wall-clock kernel time
  - `No Eligible`
  - `Eligible Warps/Scheduler`
  - `Achieved Occupancy`
  - `SM Busy`
- Since DRAM throughput is low relative to peak in both kernels, do not prioritize DRAM tuning first.
