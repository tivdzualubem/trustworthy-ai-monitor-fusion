# Missed-Harm Prevalence Estimation

## Overview

This research examines estimation of the proportion of harmful assistant responses **allowed** by a frozen post-generation safety monitor,

\[A=\Pr(Y=1,D=0).\]

The analysis combines retrospective monitor outcomes, independent response-level human reviews, cost-aware sampling experiments, and statistical assessments of label error and confidence-interval coverage. The primary endpoint is missed-harm prevalence; false-negative rate is secondary.

**Data scope:** development measurements and synthetic experiments. The 96-case human-review study is not an independently verified gold-standard dataset. Published MultiPPI and AM-PPI are represented by explicitly restricted algorithmic components, not complete author-code reproductions.

## Contents

| Directory | Contents |
|---|---|
| [`paper/`](paper/) | Technical PDF, LaTeX source and vector figures |
| [`analysis/phase11/`](analysis/phase11/) | Cost-matched estimator benchmark (synthetic reviewer labels, real review-time resampling) |
| [`analysis/phase12b/`](analysis/phase12b/) | Latency-based cost sensitivity from historical CSV measurements |
| [`analysis/phase12b_audit/`](analysis/phase12b_audit/) | Aggregate cost and estimator cross-check |
| [`analysis/phase13/`](analysis/phase13/) | Support-clipped intervals, confidence-interval diagnostics and method-identity tests |
| [`analysis/phase14/`](analysis/phase14/) | Numerically certified restricted allocation solver and finite-frame interval utilities |
| [`analysis/phase15/`](analysis/phase15/) | Private-reference study prototype: sampling, blinding and adjudication tools (not a completed review) |
| [`analysis/phase8b_wrapper/`](analysis/phase8b_wrapper/) | Synthetic robustness audit wrapper; full rerun needs an earlier local experiment module |
| [`analysis/phase10_evidence/`](analysis/phase10_evidence/) | Aggregate-only evidence checks |
| [`results/`](results/) | Machine-readable aggregate outcomes and timing provenance |
| [`docs/`](docs/) | Experimental index, method limits, reference-study design and reproducibility instructions |

Historical experiment numbers are retained **only for code and result traceability**, not as report section titles.

## Reproducibility

Install Python dependencies (`numpy`, `scipy`), then run:

```bash
bash verify_local.sh
```

The 41 core tests run with synthetic fixtures and aggregate data. Seven additional standalone tests cover latency provenance and population-robustness code. The older evidence-reconciliation tests require archived local aggregate ZIPs documented in `analysis/phase10_evidence/INPUTS_REQUIRED.md`; their absence is intentional. Regenerating cost-matched simulation outputs using measured review-time distributions requires the original **private** reviewer SQLite database, which is deliberately excluded. The repository contains no raw review cases, individual judgments, original case identifiers or private policy files.

**Current evidence:** a 1,687-response historical development geometry; 96 multiply reviewed cases (336 assessments); two synthetic benchmark population geometries (1,687 and 5,000); measured historical Koala and Qwen mean latencies; 1,200 optimizer certificate checks. See the PDF for detailed methods, results and limitations.

## Interpretation

Under synthetic perfect labeling, proxy-assisted stratification can improve point-estimation RMSE. With simulated reviewer errors, the advantage does not consistently persist. More sophisticated estimators do not resolve unknown systematic rating error. Existing results cannot identify latent true harmfulness in fresh deployment traffic without further assumptions and validation.

No new prospective reviewer labels have been collected. The current sampling prototype covers allowed cases only; the two-decision-stratum protocol is specified in [`docs/prospective_study_design.md`](docs/prospective_study_design.md) and requires separate implementation and authorization before field collection.
