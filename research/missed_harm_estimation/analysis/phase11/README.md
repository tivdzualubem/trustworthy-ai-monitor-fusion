# Phase 11 — Cost-ledgered published-method **components** comparison

**Date:** 2026-11-12 | **Status:** Runnable research experiment (synthetic latent outcomes + actual reviewer-time resampling when a private SQLite DB is supplied). **Not yet publication-ready; not the research review report.**

This package takes the next step requested by Prof. Bader without introducing fake gold labels or a new R2-review branch. It implements a single controlled synthetic benchmark for the original primary estimand, **A = P(Y=1, D=0)**. D is the frozen simulated allowed/blocked decision; Y is latent response-level harmfulness under a frozen safety policy. The reviewer endpoint Z is NOT assumed to equal Y under noise. Source/decision counts 1,687/1,557 are a geometry, not a real deployment experiment.

## What's actually implemented

| Method | Mathematical component implemented | Status |
|---|---|---|
| Whole-pool SRS | IID Bernoulli mean of `Z * 1(D=0)`; exact Clopper–Pearson CI | Valid under specified iid reviewer endpoint |
| Allowed-only SRS | IID mean on `D=0` with known allowed fraction and exact CP CI | Strong simple baseline |
| PPI++ tuned | `q [ λ mean(f0 over allowed) + mean(Z−λf0 on separate sample)]`; pilot-tuned λ charged, conditional Hoeffding CI | Mean-specialization |
| StratPPI | Two proxy-defined allowed strata; stratum-specific pilot-tuned λ and pilot-Neyman label allocation; Bonferroni-Hoeffding CI | **Mean-specialization**, not full paper generality |
| MultiPPI subset-GLS | All 7 nonempty subsets of Y/cheap/expensive; pilot covariance; constrained information-allocation optimization; integral rounding and independent subset samples; variance/CI + unbiasedness check | **Paper-grounded component**, no official solver replication or optimality certificate |
| Active AIPW restricted | Cheap-score-driven expensive-proxy subset, residual-adaptive with-replacement importance sampling, unbiased AIPW mean, conservative CI | **NOT faithful full AM-PPI** (missing paper's coupled Eq. 5/7/8/9 fixed-point, Bernoulli deployment) |

This is not presented as a win over **full** published StratPPI, MultiPPI, or AM-PPI. The methodology fidelity is explicitly recorded in the aggregate output.

## Cost and privacy contracts

- `--db` opens the user's local `cases_private.sqlite` in SQLite read-only mode and reads **only** `active_seconds` from all **336** ratings. It never selects or exports case text, reviewer credentials, vote labels or case IDs. The 336 observed active durations are resampled for simulated review actions; costs are not independently measured on new study cases.
- Calibration time and expensive predictor calibration queries are **charged**. The local pilot data are **not reused** as evaluation labels; that is conservative and avoids tuning-label leakage. Repeated labels count separately if acquired.
- The budget is in **human-active-second-equivalent normalized units**, not a physically measured single currency. A cheap proxy call defaults to `0.08` normalized units and an expensive call to `2.5`; **these are explicit assumptions, NOT actual measured model inference cost**. Full human-active seconds and per-proxy call counts are reported separately. Do not claim fully empirical economic cost parity until actual GPU/CPU inference cost, monetary conversion, and utilization are measured.
- The benchmark did not incur expert adjudication, annotation training, hardware acquisition or new response generation. Thus those components are `NOT MEASURED / NOT INCURRED`, not zero observed costs for a real study.
- A hard budget is enforced in every trial by requiring room for the **maximum observed** review duration before starting each human action. Released unused reserves remain available. Empirical duration resampling is assumed independent of case difficulty/label; this is an explicit limitation.
- Since Y is not observed in the real data, **coverage for Y in synthetic runs is diagnostic**, not inferential. The explicit finite-sample conservative intervals target the **simulated reviewer expectation Z**, conditioned on the calibration pilot, score/policy model, duration independence and iid-with-replacement acquisition. No real safety certificate is claimed.
- The original 96 reviewer cases and 336 judgments are left untouched. No R2-only verification is part of Phase 11.

## One WSL execution after downloading ZIP into Windows Downloads

Use the command provided in the ChatGPT response. The following is the direct Python invocation once inside this extracted directory:

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
"$PY" -m unittest -q test_phase11_benchmark.py
"$PY" phase11_benchmark.py \
  --db "$PRIVATE_DIR/cases_private.sqlite" \
  --geometry development_1687 \
  --populations 8 --repetitions 25 \
  --budget 4800 \
  --cheap-unit-equivalent 0.08 --expensive-unit-equivalent 2.5 \
  --out-dir "$PRIVATE_DIR/phase11_benchmark_20261112/development"
"$PY" phase11_benchmark.py \
  --db "$PRIVATE_DIR/cases_private.sqlite" \
  --geometry phase6_5000 \
  --populations 8 --repetitions 25 \
  --budget 4800 \
  --cheap-unit-equivalent 0.08 --expensive-unit-equivalent 2.5 \
  --out-dir "$PRIVATE_DIR/phase11_benchmark_20261112/phase6"
```

The output directory must **not already exist**; accidental overwrites are refused. Each geometry writes `phase11_aggregate.json` and `phase11_aggregate.md`. All values are aggregate synthetic trials and real time **statistics**, no raw private database data.

## Verification

Ten unit tests cover population dimensions, known Y versus reviewer Z, complete calibration/proxy cost ledger, no zero-cost human oracle, the λ pilot separation, 7-subset MultiPPI allocation feasibility, hard budgets, paired population bootstrap, HH correction identity, SQLite read-only duration access, and response-text privacy invariants. A full **illustrative-duration** 8×25 run has also completed with no hard-budget violations. Its outputs are bundled under `ILLUSTRATIVE_REFERENCE_ONLY/`, not as validated real reviewer-cost findings.

## Relevant published work

- Fisch et al. 2024, StratPPI: https://arxiv.org/abs/2406.04291
- Cowen-Breen et al. 2026, MultiPPI: https://arxiv.org/abs/2603.27414 (Section 4 / Eq. 7)
- Brawand et al. 2026, AM-PPI: https://arxiv.org/abs/2605.08429 (Eqs. 1, 5, 7–9)
- Feng et al. 2026, Noisy but Valid: https://arxiv.org/abs/2601.20913

## After Phase 11

This run closes a substantive *methods-and-budget simulation comparison*, but **cannot itself close** (i) exact author-reproduced full AM-PPI, (ii) measured inference cost parity, (iii) independent expert gold-reference outcomes, or (iv) prospective post-generation/pre-delivery native-monitor validation. These requirements must be addressed with genuine source code, measured inference, and (where authorized) new blinded human evidence. We do not silently mark them complete.

**Do not push code or issue the final research lead report based on this preliminary synthetic benchmark alone.**
