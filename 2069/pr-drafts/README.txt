GH-2069 PR body drafts
Prepared 2026-10-10. These files are reviewable drafts, not published PRs.

1. Mixed precision
Title: Support mixed-precision BSR matrix-vector products (GH-2069)
Head: fusheng-ji:feat/bsr-mv-mixed-precision
Intended base: NVIDIA/warp:main
Body: mixed-precision.md
Code commit: 4db534d616d5ff16b31c8b851d93690dfc296fd8

2. Row packing
Title: Add optional row-packed CUDA BSR products (GH-2069)
Head: fusheng-ji:perf/bsr-mv-row-packed
Current dependency: feat/bsr-mv-mixed-precision
Intended base: NVIDIA/warp:main after contribution 1 merges and this branch is rebased
Body: row-packed.md
Incremental code commit: 589fd1216f21ef500aa0d7bfe2e4dc8d6c3086ce

Both bodies follow the repository PR template and use Part of #2069.
Do not label the second branch independent of the first: it has one incremental
commit above the mixed-precision branch and currently two commits above main.
Replace the dependency branch link with the actual PR link when available.

Evidence links and inline images are pinned to asset commit
b43287be27bc7980ae65668bc8ce18fbd8ab9297. Publication fields in those historical
validation records describe the time of validation, before branch upload.
Fresh split-specific tests are distinguished from the earlier timing runs.

Before opening upstream PRs, refresh the main reference and complete the open-PR
overlap search. The current drafts leave that checkbox unchecked. The complete
contribution-guidelines checkbox is also unchecked because external contribution
coordination is pending. No maintainer assignment, concrete downstream use, or
approval is implied. No issue comments were posted as part of this preparation.

The publication bundle excludes raw workstation logs and private paths. Public
author attribution and DCO sign-off are retained; no AI co-author trailer is added.
