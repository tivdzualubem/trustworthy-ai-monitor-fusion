# Trustworthy AI — Phase 12B interpretation and integrity audit

**Date:** 2026-11-12. **Scope:** uploaded synthetic cost sensitivity aggregate results; no private case data and no project repository modifications.

This archive contains:
- `PHASE12B_RESEARCH_DECISION.md`: detailed scientific interpretation, paired-bootstrap comparisons, interval-width audit, and exact remaining requirements.
- `phase12b_verified_crosscheck.json`: machine-readable derived comparisons.
- `audit_phase12b.py`: read-only reproducibility script with assertions on counts, costs, provenance, and unchanged SRS results.

To reproduce with the same five input JSONs downloaded to `/path/to/inputs`:

```bash
python3 audit_phase12b.py \
  --phase11-development /path/to/inputs/phase11_development_20261112.json \
  --phase11-phase6 /path/to/inputs/phase11_phase6_20261112.json \
  --phase12b-development /path/to/inputs/phase12b_development_20261112.json \
  --phase12b-phase6 /path/to/inputs/phase12b_phase6_20261112.json \
  --provenance /path/to/inputs/phase12b_cost_provenance.json \
  --out-dir /tmp/phase12b_verified
```

Nothing here establishes full AM-PPI / MultiPPI fidelity, real latent Y coverage, or measured marginal economic price parity. No GitHub push or research review report was performed.
