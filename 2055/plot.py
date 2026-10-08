# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
rows = []
for phase in ('before', 'after'):
    with (root / f'{phase}.csv').open() as f:
        rows.extend(csv.DictReader(f))


def select(phase, device, dtype):
    return [r for r in rows if (r['phase'], r['device'], r['dtype']) == (phase, device, dtype)]


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': '#f8fafc',
                     'axes.facecolor': '#ffffff', 'savefig.facecolor': '#f8fafc'})
fig, axes = plt.subplots(1, 3, figsize=(16, 6.3))
fig.subplots_adjust(left=.06, right=.985, top=.79, bottom=.32, wspace=.32)
fig.suptitle('FP16 SVD3: before / after numerical validation', x=.06, y=.96,
             ha='left', fontsize=21, fontweight='bold', color='#12243a')
fig.text(.06, .875, 'Unmodified base e207ab67 vs. focused half-precision fix  |  CPU + NVIDIA B200 / CUDA 13.0',
         fontsize=11, color='#536477')

ax = axes[0]
groups = [select(p, d, 'float16') for d in ('CPU', 'CUDA') for p in ('before', 'after')]
passed = np.array([sum(r['passed'] == 'True' for r in g) for g in groups])
nonfinite = np.array([sum(r['finite'] != 'True' for r in g) for g in groups])
inaccurate = np.array([len(g) for g in groups]) - passed - nonfinite
x = np.arange(4)
ax.bar(x, passed, color='#0f897b', label='Pass', width=.62)
ax.bar(x, inaccurate, bottom=passed, color='#e5a13b', label='Finite, fails accuracy', width=.62)
ax.bar(x, nonfinite, bottom=passed+inaccurate, color='#c54e59', label='Nonfinite factor', width=.62)
for i, count in enumerate(passed):
    ax.text(i, 106, f'{count}/102', ha='center', fontsize=10, fontweight='bold')
ax.set(xticks=x, xticklabels=['Before\nCPU', 'After\nCPU', 'Before\nCUDA', 'After\nCUDA'],
       ylim=(0, 116), ylabel='Matrix evaluations (both calling forms)', title='A   FP16 outcomes')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.21), frameon=False, fontsize=9, ncol=1)

ax = axes[1]
colors = {'CPU': '#2167aa', 'CUDA': '#cc7243'}
exponents = [-20, -10, 0, 10, 14]
for d, marker in (('CPU', 'o'), ('CUDA', 'x')):
    group = select('after', d, 'float16')
    values = [max(float(r['reconstruction_budget_ratio']) for r in group
                  if r['matrix'] != 'zero' and int(r['exponent']) == e) for e in exponents]
    ax.plot(range(5), values, marker=marker, color=colors[d], label=d, linewidth=1.8,
            markersize=8, linestyle='-' if d == 'CPU' else '--')
ax.axhline(1, color='#c54e59', linestyle=':', linewidth=1.4, label='Acceptance limit')
ax.set(xticks=range(5), xticklabels=[f'$2^{{{e}}}$' for e in exponents], ylim=(0, 1.08),
       xlabel='Input scale', ylabel='Max reconstruction error / allowed error',
       title='B   FP16 after: error margin')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.21), frameon=False, fontsize=9, ncol=2)
ax.grid(axis='y', alpha=.18)

ax = axes[2]
dtypes = ['float16', 'float32', 'float64']
for d, offset in (('CPU', -.16), ('CUDA', .16)):
    values = [max(max(float(r['orth_u']), float(r['orth_v'])) for r in select('after', d, t))
              for t in dtypes]
    ax.bar(np.arange(3)+offset, values, width=.28, color=colors[d], label=d)
for i, tol in enumerate((5e-3, 1e-5, 1e-12)):
    ax.plot([i-.34, i+.34], [tol, tol], color='#c54e59', linestyle=':', linewidth=1.7,
            label='Acceptance limit' if i == 0 else None)
ax.set(yscale='log', xticks=range(3), xticklabels=['FP16', 'FP32', 'FP64'],
       ylim=(5e-17, 1e-2), ylabel='Max entrywise orthogonality error',
       title='C   After: U / V orthogonality')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.21), frameon=False, fontsize=9, ncol=2)
ax.grid(axis='y', alpha=.18)

fig.text(.06, .035, 'Metrics use returned factors promoted to NumPy float64. Allowed reconstruction error = '
         'tolerance × max|A| + 2 × smallest subnormal.\n'
         'No timings or speedup claims. Full data: 532 evaluations per version; FP16 includes subnormal-scale inputs.',
         fontsize=9, color='#536477', linespacing=1.6)
fig.savefig(root / 'svd3-validation.png', dpi=160, metadata={'Software': 'Matplotlib'})
fig.savefig(root / 'svd3-validation.svg', metadata={'Date': None, 'Creator': 'Matplotlib'})
plt.close(fig)

summary = []
for d in ('CPU', 'CUDA'):
    for t in dtypes:
        before, after = select('before', d, t), select('after', d, t)
        summary.append(dict(device=d, dtype=t, cases=len(after),
                            before_nonfinite=sum(r['finite'] != 'True' for r in before),
                            before_failed=sum(r['passed'] != 'True' for r in before),
                            after_failed=sum(r['passed'] != 'True' for r in after),
                            max_scaled_reconstruction=max(float(r['reconstruction_scaled']) for r in after),
                            max_error_budget_ratio=max(float(r['reconstruction_budget_ratio']) for r in after),
                            max_orthogonality=max(max(float(r['orth_u']), float(r['orth_v'])) for r in after)))
with (root / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
for r in summary:
    print(r)
