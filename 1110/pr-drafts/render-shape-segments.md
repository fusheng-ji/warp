## What problem are you trying to solve?

Closes #1110.

At the tested baseline, `OpenGLRenderer.render_*()` does not expose the existing primitive mesh generators' segment counts. Debug scenes with many capsules therefore use the default mesh density. This proposal lets callers choose the density and change an existing named primitive's resolution between frames.

This work comes from an independent renderer correctness and performance investigation. I do not have a downstream application or production deployment to report. [Outside-implementation guidance was requested in #1110](https://github.com/NVIDIA/warp/issues/1110#issuecomment-6297170911); there is no maintainer reply at this snapshot, and no assignment or approval is claimed. This document is proposed PR text; no feature PR has been opened.

## What does this change?

- Add keyword-only `segments` to sphere, capsule, cylinder, cone and arrow rendering, preserving existing positional arguments and the default of 32. Sphere latitude/longitude and arrow base/cap share the supplied resolution.
- Accept Python/NumPy integers, reject booleans and nonintegers, and reject values below 3 before changing renderer state.
- Cache geometry by resolution and rebind same-name instances when resolution changes, preserving identity, parent, scale and custom ID while updating poses and colors through the existing flow. Switching back reuses the previous mesh; template calls create/reuse geometry without adding instances.
- Add regression tests, method/guide documentation and `1110.added.md`.

With an existing renderer, callers can change resolution over successive frames:

```python
for frame, segments in enumerate((32, 8, 32)):
    renderer.begin_frame(frame / 60.0)
    renderer.render_capsule(
        "debug_capsule",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        radius=0.5,
        half_height=0.6,
        segments=segments,
    )
    renderer.end_frame()
```

The scope covers these five OpenGL primitives. Point clouds, lines, USD rendering, visibility/deletion behavior and mesh-generation algorithms remain outside this proposal. The separate CPU/OpenGL buffer fix is draft [#2121](https://github.com/NVIDIA/warp/pull/2121).

## Existing pull requests

- [x] I searched open pull requests for duplicate or overlapping work.
- Related pull requests: #2121 fixes CPU buffer mappings and does not implement primitive resolution. Searches for open PRs mentioning segments, primitive resolution and #1110 found no duplicate implementation. The [search snapshot](duplicate-search.json) records the queries.

## Validation

Frozen feature source: **`cb86abb75d123876d80c6832d8d360f4cda0cc10`**, one DCO-signed commit based on official main **`0382a97e2c72911c1f6bba45274e12833a19d8e2`**. The current official main has advanced; this proposal has not been rebased or retested on that newer main. The [Git bundle](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/source.bundle) restores the exact measured source without depending on an unpublished branch.

- Original regression validation: the four added unittest methods fail on the unmodified baseline because `segments` is unavailable. All **14 context-free renderer test methods** pass on the committed feature. Tests cover five primitives, default/32/8/3/NumPy-integer values, invalid values before state changes, sharing, templates, cached/uncached switching, stable identities and instance attributes.
- Original checks: an independent standard native build, **30 wireframe B200 CUDA/OpenGL frames**, strict HTML documentation and applicable formatting checks passed. Documentation/source hashes connect the pre-commit checks to the committed source. These are original acceptance records, not new runs for this publication.
- New primary correctness validation: three independent `overflow` allocations on **three distinct nodes**, each exclusively allocated one full B200 GPU, passed **120 filled frames** and **30 exact restored images**. Each tests five shapes, cached/uncached targets and `32 → 8 → 3 → 32`; tiled selection, both objects' visibility, identity/membership/mapping, transforms and GL errors are checked.
- New primary timing: **6,840 completed frames in 270 blocks**. Jobs **1144767, 1144768, 1144769**, with case-order seeds **1121, 1132, 1143**, all completed with exit **0:0**. Each uses a private cache, one visible CUDA device, disabled MIG and a strict native CUDA/EGL interop probe; no GPU process was present before the probe.
- Environment: NVIDIA B200, Linux x86_64, NVIDIA driver **580.126.20**, EGL/OpenGL **3.3**, Python **3.12.13**, Warp **1.19.0.dev1**, NumPy **2.5.3**, Pyglet **2.1.16**; reused source-matched CUDA 13.0 build artifacts. Source, library hashes, benchmark script, versions and thread settings match across all three new jobs. No new native build or GPU experiment was needed to publish the results.

[Full report and provenance](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/README.md) · [Original acceptance](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/original/validation.txt) · [Three-job report](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/independent-summary/REPORT.txt) · [Scheduler results](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/independent-summary/ACCOUNTING.csv) · [Reproduction instructions](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/REPRODUCE.md) · [Data recomputation](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/recomputation.json) · [Checksums](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/SHA256SUMS)

### Filled correctness and minimum resolution

![Five primitives at 32, 8, 3 and restored 32 segments; job 1144767 captures](https://raw.githubusercontent.com/fusheng-ji/warp/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/jobs/1144767/filled_comparison.png)

Minimum resolution 3 is supported and remains visible, but its geometry and shading are substantially coarser. Lower-resolution timings compare different visual fidelity settings.

### Static filled rendering

Values are **ms/frame: median of three job summaries [minimum, maximum]**. A job summary is the median of five block medians. Every frame waits for GL completion.

| Capsules | 32 segments | 16 segments | 8 segments | 3 segments |
|---:|---:|---:|---:|---:|
| 64 | 0.610 [0.607, 0.610] | 0.483 [0.480, 0.484] | 0.429 [0.429, 0.430] | 0.412 [0.410, 0.413] |
| 512 | 1.757 [1.756, 1.759] | 0.829 [0.829, 0.831] | 0.544 [0.542, 0.545] | 0.428 [0.428, 0.431] |
| 4096 | 10.915 [10.906, 10.922] | 3.338 [3.333, 3.341] | 1.385 [1.383, 1.387] | 0.627 [0.625, 0.628] |

### All-instance switching on every frame

The same latency statistic includes `render_capsule()` calls for every instance, instance-buffer rebuilding/upload work and completed drawing while alternating cached resolutions.

| Capsules | Alternating 32/8 | Alternating 32/3 |
|---:|---:|---:|
| 64 | 1.700 [1.593, 1.718] | 1.628 [1.625, 1.683] |
| 512 | 4.458 [4.435, 4.538] | 4.136 [4.054, 4.520] |
| 4096 | 26.271 [25.049, 27.905] | 24.848 [24.748, 26.560] |

The dynamic costs are retained: at 4096 capsules, alternating 32/8 takes **26.271 ms/frame [25.049, 27.905]**, compared with a static 8-segment scene at **1.385 ms/frame [1.383, 1.387]**. Geometry caching does not eliminate per-frame switching work; no automatic LOAD policy or switching optimization is claimed.

![Static latency, paired latency ratios and dynamic-switch latency across three allocations](https://raw.githubusercontent.com/fusheng-ji/warp/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/independent-summary/across_jobs.png)

[Full-precision summary CSV](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/independent-summary/across_jobs.csv) · [Per-job summaries](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/independent-summary/per_job_summary.csv) · [Raw frames and logs by job](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/jobs/) · [Recomputable Markdown tables](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/TABLES.md)

The timed scenes use filled polygons at 1280x720, vsync off and no grid/sky/axis/overlay. Capsule size shrinks with instance count to keep a fixed scene extent. Mesh creation, initial upload, eight warmup frames per block, readback and query-result retrieval are excluded. Each job runs five randomized repeats, with 32 measured static frames or 12 measured dynamic frames per block. Static frames omit per-object API calls.

These are **three allocations**, not 6,840 independent statistical samples. The bars are observed ranges, not confidence intervals. Nodes and their other GPUs can be shared; CPU/node activity, GPU variation and thermal state can affect results, and no clocks were locked. Slow samples remain in the raw data. Frame-latency reciprocals are not application or display FPS, and results do not establish a speedup against a separately timed upstream checkout.

### Separate historical records and failures

The original 30 wireframe frames, shared-GPU 40 filled/2280 timed frames and first exclusive job 1143896's 40 filled/2280 timed frames remain separate historical datasets. They are excluded from the new three-job totals above. The earlier feature harness error was corrected by supplying parent transforms without changing the product or pixel tolerance. Slurm device-selection jobs 1143879 and 1143894 failed before rendering/timing; their diagnostics are retained in the [attempt history](https://github.com/fusheng-ji/warp/blob/036fceb1a7935b8787207ad9bbc4b97b95b3b498/1110/validation/cb86abb7/overflow/ATTEMPTS.txt). Older CPU/feature combined validation belongs to a different combined source and supplies no new CPU validation here.

Assets are pinned to **`036fceb1a7935b8787207ad9bbc4b97b95b3b498`**. Collection-time “local only” statements inside retained files describe their historical snapshots. Public assets exclude raw private scheduler logs and preserve anonymous node labels, public noreply attribution and DCO.

## Checklist

- [ ] I have read and followed the [Contributing Guidelines](https://github.com/NVIDIA/warp/blob/main/CONTRIBUTING.md). Read; outside-implementation coordination is pending, with no maintainer approval implied.
- [x] This pull request does not bundle unrelated changes unless a maintainer agreed to the broader scope.
- [ ] My branch is up to date with `main` and contains one commit, unless a maintainer agreed to keep the change as multiple commits. The measured source is one commit on the explicitly recorded older baseline.
- [x] I included any required [changelog fragments](https://github.com/NVIDIA/warp/blob/main/changelog/README.md) and used a `+` prefix only for changes that do not require a GitHub issue.
