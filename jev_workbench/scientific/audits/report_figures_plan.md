# Final-result figure pipeline — scoped implementation plan

Scope: create scientific/report_figures.py and synthetic-only tests. No real test predictions, API calls, development results, old article figures or publication actions are read or used.

Figure contract: Python/matplotlib; quantitative evidence figures for an experiment report. Formal numbers must come only from a complete final_analysis.json produced by analyze.py. Record its SHA256, the frozen bundle SHA256 and plotting-source SHA256 in a manifest and per-figure sidecars. Top-level and every primary/round score must be complete, without pending/provisional records. All model conditions and planned failure outcomes remain in the figures and source data.

Planned evidence:

- Per task all-model, all-planned Accuracy with analysis 95% bootstrap CI. RAG uses pair-weighted points and query-cluster CI, not a CI assigned to equal-query mean points.
- Per task all-model Macro-F1 with exploratory CI; no significance stars or F1 significance language. Record fixed label universe and unsupported-label policy.
- Primary-repeat valid vs failure counts/rates with full planned denominator and failure types.
- Per task repeated-round Accuracy/Macro-F1/valid rate; descriptive agreement and conditional valid-pair denominators; repeats do not increase independent n, deterministic one-round models remain one round.
- All-repeat operational latency per model, separating logical inclusive-retry, valid, failure and HTTP attempt scopes. Retain missing counts; local and remote scales are never combined into a ratio ranking.
- Primary-repeat JEV, Gemini and dev-selected best-local confusion matrices, keeping every true label and NO_VALID_OUTPUT failure column.

TDD sequence: write synthetic complete-fixture tests first, witness failures for absent implementation; implement completion/integrity gates; export source data and PNG/SVG plus QA PDF; verify manifests, NO_VALID_OUTPUT counts and refusal cases. All exported fixture images are synthetic test artifacts, explicitly marked, outside formal-result directories. No final-model figures exist until valid final data are available.

QA: saved Python backend; editable SVG, >=5 pt text, restrained palette; all plots use single axes to avoid unverified multi-panel alignment. Run source preflight and rendered PDF text/collision QA on synthetic export, inspect preview, and preserve findings. Actual final export still requires its own visual QA.

## Implementation and verification result

- Implemented report_figures.py; only complete final-analysis schema is accepted. Integrity gates run before any output directory is created. Whole dataset/model/round denominator checks preserve failures and NO_VALID_OUTPUT; source hash, frozen bundle hash, and plotting-source hash are recorded per figure.
- TDD: initial 10 tests visibly failed because the module was absent; implementation turned them green. Added top-level pending, inconsistent Macro-F1 and latency coverage cases; top-level pending visibly failed and was repaired. Final synthetic-only suite: 13 passed.
- Synthetic fixture export produces 11 single-axes figure families for one task, PNG + editable SVG + QA PDF, sidecar manifest, exact source-data JSON and aggregate manifest. Fixture exports live only under /tmp and have an explicit SYNTHETIC CONTRACT TEST / NOT MODEL RESULTS watermark and purpose.
- Static source preflight: 17 PASS, 4 WARN, zero FAIL. Report-specific warnings: PNG 300 dpi (not TIFF/600 dpi journal production), dynamic report dimensions, and descriptive repeat/latency summaries with no inferential uncertainty. Accuracy and exploratory F1 do encode the supplied 95% intervals.
- All 11 synthetic QA PDFs pass the 5 pt text-size floor. Rendered collision auditing is NOT AUDITABLE: system and bundled Python lack PyMuPDF. No collision validation claim is made; all figure manifests state rendered/final visual review remains required. Selected synthetic Accuracy, latency, confusion and stability previews were visually inspected.
- No formal data, formal model results, development data, source predictions or API calls were used. No figure is ready for a formal-results article until final analysis exists and its own rendered/visual QA is done.
