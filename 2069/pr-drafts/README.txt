GH-2069 PR body drafts
Prepared 2026-10-10. The first contribution is published as PR #2072;
the second body remains a draft for publication.

1. Mixed precision
Published PR: https://github.com/NVIDIA/warp/pull/2072
Title: Support mixed-precision BSR matrix-vector products (GH-2069)
Head: fusheng-ji:feat/bsr-mv-mixed-precision
Intended base: NVIDIA/warp:main
Body: mixed-precision.md
Code commit: 4db534d616d5ff16b31c8b851d93690dfc296fd8

2. Row packing
Title: Add optional row-packed CUDA BSR products (GH-2069)
Head: fusheng-ji:perf/bsr-mv-row-packed
Current dependency: https://github.com/NVIDIA/warp/pull/2072
Intended base: NVIDIA/warp:main after contribution 1 merges and this branch is rebased
Body: row-packed.md
Incremental code commit: 589fd1216f21ef500aa0d7bfe2e4dc8d6c3086ce

Both bodies follow the repository PR template and use Part of #2069.
Do not label the second branch independent of the first: it has one incremental
commit above the mixed-precision branch and currently two commits above main.
The row-packed body now links to #2072 and a commit-pinned incremental diff.
Keep the second PR as a draft until #2072 merges and this branch is rebased.

Evidence links and inline images are pinned to asset commit
b43287be27bc7980ae65668bc8ce18fbd8ab9297. Publication fields in those historical
validation records describe the time of validation, before branch upload.
Fresh split-specific tests are distinguished from the earlier timing runs.

The open-PR search checkbox is checked in both drafts. On 2026-10-10, the public
GitHub API returned all 60 open upstream PRs in one page. Their titles and bodies
were searched for GH-2069, BSR, mixed precision, and row packing. Sparse changes
in related PRs #1791 and #1975 were inspected; no direct duplicate was found.
Official main was rechecked and still matched c47fe4c8.

The stored drafts leave the complete contribution-guidelines checkbox unchecked
because external contribution coordination is pending. The published #2072 body
checks it while still stating coordination is pending; that inconsistency was
reported during review, and the published body was not edited in this update.
The row-packed draft's single-commit/main checkbox remains unchecked because it
is stacked on #2072.
No maintainer assignment, concrete downstream use, or approval is implied.
No issue comments were posted as part of this preparation.

The publication bundle excludes raw workstation logs and private paths. Public
author attribution and DCO sign-off are retained; no AI co-author trailer is added.
