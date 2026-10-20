# Independent allocation results

Source: `cb86abb7`. Jobs: 1144767, 1144768, 1144769 on three distinct nodes, one exclusive B200 GPU each.

Values are ms/frame: median of three job summaries [minimum, maximum]. Each job summary is the median of five block medians. Ranges are observations, not confidence intervals.

## Static filled rendering

| Capsules | 32 segments | 16 segments | 8 segments | 3 segments |
|---:|---:|---:|---:|---:|
| 64 | 0.610 [0.607, 0.610] | 0.483 [0.480, 0.484] | 0.429 [0.429, 0.430] | 0.412 [0.410, 0.413] |
| 512 | 1.757 [1.756, 1.759] | 0.829 [0.829, 0.831] | 0.544 [0.542, 0.545] | 0.428 [0.428, 0.431] |
| 4096 | 10.915 [10.906, 10.922] | 3.338 [3.333, 3.341] | 1.385 [1.383, 1.387] | 0.627 [0.625, 0.628] |

## All-instance switching on every frame

| Capsules | Alternating 32/8 | Alternating 32/3 |
|---:|---:|---:|
| 64 | 1.700 [1.593, 1.718] | 1.628 [1.625, 1.683] |
| 512 | 4.458 [4.435, 4.538] | 4.136 [4.054, 4.520] |
| 4096 | 26.271 [25.049, 27.905] | 24.848 [24.748, 26.560] |

These 120 filled frames, 30 exact restored images, 6840 timed frames and 270 blocks exclude all previous allocations and shared-GPU runs. The statistical units here are three allocations, not 6840 independent samples.

Lower resolution changes visual fidelity. Timings wait for GL completion; they are not application/display FPS. Nodes and other GPUs can be shared; no clocks were locked. Dynamic switching includes all per-object calls and rebuilding/upload work.

[Full-precision CSV](overflow/independent-summary/across_jobs.csv) · [Per-job CSV](overflow/independent-summary/per_job_summary.csv) · [Raw frames by job](overflow/jobs/)
