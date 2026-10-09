# Stage-A Phase 3 and response-audit Phase 4 exploratory release (9 Oct 2026)

**Branch:** `stage-a-paired-transport-20260921`. This release adds evidence and executable analysis; it does **not** revise the Oct 7 frozen scoring inputs, the 0/4 readout-transfer screening decisions, or prior model checkpoint settings.

## Scope and key distinctions

- **Phase 3** uses a 38-pair/76-case prompt-only `case_type` proxy. Its balanced accuracies are NOT response-level safety/FNR rates.
- **Phase 3C** includes 38 new same-topic hard-benign controls, one-researcher signoff on candidate text and a separate AI-assisted native-policy review. Eighteen of 304 new policy judgments remain `uncertain`; no independent multi-rater agreement is claimed.
- **Phase 4** reuses exactly 1,687 allowed *development* prompt-response cases per historical classifier setup, with the historical response label `y`. `base_prediction` denotes a cross-fitted development classifier, **not** a frozen native guard's deployed decision. These are simulation/feasibility results, not live FNR or prospective confirmation.
- Neither protected final-test nor shift data were accessed by these scripts; their boundaries depend on the source repository and whitelisted existing development artifacts.

## Run sequence (from repository root, Python virtual environment)

The original Phase 3 depends on the frozen Stage-A `results/stage_a_readout_transfer_kill_v1/features/` NPZ files. Phase 4 needs existing development-only Parquet and fold CSVs in `reports/decision_value_real_data/`, which are **not** included in this release.

```bash
.venv/bin/python scripts/test_stage_a_phase3_cpu.py
.venv/bin/python scripts/stage_a_phase3_cpu.py
.venv/bin/python scripts/stage_a_phase3c_hard_benign.py
.venv/bin/python scripts/phase4_dev_response_audit_pilot.py
.venv/bin/python scripts/phase4b_active_testing_lure.py
.venv/bin/python scripts/phase4c_monitor_assisted_audit.py
.venv/bin/python scripts/phase4d_optional_score_falsification.py
```

**CAUTION:** these original scripts were written to refuse overwriting some existing versioned result folders and may copy ZIPs to Windows Downloads. To reproduce, run in a fresh checkout/worktree with the frozen input data present or move only the newly generated result directories to a fresh copy; **do not** delete or overwrite the committed outputs to force a rerun. Dependencies and exact original environment are recorded in output manifests, including scikit-learn and numerical libraries. This package preserves the scripts as executed, rather than editing them after observing results.

## Output / evidence map

| Versioned path | Evidence |
|---|---|
| `results/stage_a_phase3_shortcut_threshold_v1/` | Source-probe cutoff / word-char same-fold baselines and OOF scores |
| `results/stage_a_phase3c_hard_benign_v1/` | Fresh candidate hard-benign text and CPU lexical pilot, pre-signoff worksheets |
| `results/stage_a_phase3c_prompt_approved_v1/` | Researcher-approved text, provenance and pilot results |
| `results/stage_a_phase3c_policy_review_researcher_approved_v1/` | 304 guard-policy judgments and 38 O2 equivalence assessments, with 18 unknown policy cases |
| `results/phase4_dev_response_audit_sampling_v1/` | Random, stratified, allowed-only human audit pilot |
| `results/phase4b_active_testing_lure_v1/` | Sequential active-testing LURE benchmark and baseline comparison |
| `results/phase4c_monitor_assisted_audit_v1/` | Monitor-assisted acquisition preliminary comparison |
| `results/phase4d_optional_score_falsification_v1/` | Paired cheap/acquisition/permuted/real-score ablation and scalar errors |
| `docs/phase3_phase4_release_manifest_20261009.json` | File-level checksums of all new release materials |

## Findings and limitations

- Original source-only prompt-label lexical BACC on easy controls: 1.0 for both word and character models. Same-topic hard negatives lower within-representation BACC to 0.6711 (word) and 0.7237 (char).
- Source-probe O2 BACC after target-training-fold cutoff-only adjustment: Llama 0.6447, Shield 0.7895, Granite 0.9474, Qwen 0.9342. Earlier mean-shift of a fixed linear classifier modifies intercept only. The original screen **still fails 0/4**.
- On the development response task, the published-inspired LURE Active Testing adaptations do not consistently beat proportional stratified human audits for FNR estimation.
- In paired Phase 4D simulations, Qwen purchased scores reduce FNR RMSE at 320 optional calls / 160 human reviews from 0.0605 (permuted scores) to 0.0421 (real scores); 95% bootstrap MC interval on real-minus-permuted RMSE around [-0.0240,-0.0131]. This quantifies Monte Carlo uncertainty conditional on the historic development population, **not** deployment confidence. Qwen 320 calls incur about 511.2 s of sum-of-legacy-mean inference time; human review costs are not measured. Missed-harm prevalence advantages are less clear.

No native-guard rescoring on the new hard-benign prompts, no prospective independent response-level adjudication, no fresh model timing, no operational cost/latency guarantee, and no actual post-shift audit certificate are claimed. Proposed next research step: seek supervisor agreement on a small frozen prospective response-level sampling/monitor-purchase study, not another post-hoc adapter.

**Supervisor-facing PDF is intentionally not included in the GitHub release.** It must be sent separately.
