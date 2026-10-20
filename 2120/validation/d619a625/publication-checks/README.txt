Source-code pre-commit runs are recorded separately under logs/.
The first asset hook run identified helper-script style findings and spelling
false positives in kernel hashes, package names and generated diagnostic logs.
The helpers were corrected. The final applicable hook run covers all Python
reproduction/plot tools, handwritten reports/instructions and the asset index.
Machine-generated CSV/JSON/log records are preserved rather than spell-corrected;
checksums cover those files too. No numerical tolerance or test was relaxed.
