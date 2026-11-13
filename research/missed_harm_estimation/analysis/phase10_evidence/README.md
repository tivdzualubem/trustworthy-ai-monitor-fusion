# Internal Phase 10 Evidence-Reconciliation Bundle

This package is an **intermediate research audit**, not a final result or a research review/GitHub submission. It carries copies of five aggregate artifacts (no reviewer text, individual case records, private lookups, or credentials), an auditable reconciliation script, eight tests, an internal evidence report, and a completion protocol that spells out the exact remaining scientific requirements.

Run locally with Python 3.10+ without extra dependencies:

```bash
python3 -m unittest -v test_phase10_evidence_gate.py
python3 phase10_evidence_gate.py \
    --phase8b inputs/phase8b_population_robustness_20261112.json \
    --phase9b inputs/phase9b_audit_20261112.json \
    --phase5b1 inputs/phase5b1_blocked_allowed_baselines_v1.zip \
    --phase5b2 inputs/phase5b2_stratppi_mean_development_v1.zip \
    --phase5b3 'inputs/phase5b3_amppi_restricted_v1(1).zip' \
    --out-dir output_reproduced
```

Do not re-run in an existing output folder; it refuses overwrites. `generated/` contains the already verified outputs. We do **not** ask the user to re-run previous experiments in WSL just to read this result.

**Results and limitations:**
- Verified historical single-case Qwen allowed-harm delta; historical classification is **not** a native guard.
- Existing specific StratPPI mean specialization does not outperform allowed-only SRS at the examined 160-label budget; this is not full published StratPPI and uses unverified historical labels.
- Restricted exploratory AM-PPI overspent realized budgets in some simulations; it cannot establish hard-budget validity or published-method performance.
- Phase8B synthetic population-bootstrap comparisons are robust within the fixed error model; cannot identify actual rater errors.
- Phase9B real reviewers disagree; weighted estimates and approximate sampling CIs concern their **own outcomes**, not latent Y.

**Next priority is not R2-specific QC.** It is a single fair published-method comparison with full calibration/proxy/human/adjudication cost, followed by a well-prepared prospective native-monitor/reference-label pilot. We should publish only after the claims and scope are defensible.

**Original sources:** See SHA256 hashes in `generated/phase10_evidence_gate.md`, and the original-source references in the archived manifests. All code is new reconciliation code; no GitHub files have been altered.
