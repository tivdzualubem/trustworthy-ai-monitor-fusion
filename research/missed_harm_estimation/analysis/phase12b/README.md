# Phase 12B — measured historical proxy latency, assumed economic conversion

Run a deterministic read-only public timing-summary extraction. A second optional step reruns the existing, pre-tested Phase 11 synthetic estimator benchmark with **historical component mean latencies instead of made-up fixed prices**, under an explicitly **assumed marginal cost ratio** of GPU-second to human-review-second. This is **not fully measured economic parity**.

- `phase12b_latency_cost.py` reads **only three existing public repository CSV summaries** (never raw prompt-response text, raw case identifiers, reviewer votes, or reviewer databases); each source is SHA256 hashed for provenance.
- `--run-phase11` spawns the already tested Phase 11 script. That child reads **only review-duration observations** from the private SQLite database in read-only mode. The child produces aggregate-only synthetic simulation outcomes; no private text/labels are exported.
- The old Phase 11 unit prices were **0.08 and 2.5**. The new setting is `cheap_price = (mean Koala latency in milliseconds / 1000) × ρ` and `expensive_price = (mean Qwen latency in milliseconds / 1000) × ρ` where `ρ` is a **chosen sensitivity ratio**, NOT a measured exchange rate. There is no assumption that historical GPU runtime equals human attention or that costs on distinct hardware are automatically economically comparable.
- A distinct **363-case T4 policy timing** experiment is retained as separate corroborating evidence, not merged with the 128-case component experiment. Whole-policy latencies do not equal a single model call.
- Synthetically generated proxy predictions remain synthetic and have not been verified to reproduce the actual Koala/Qwen scores on this exact population.
- No training, calibration labor beyond recorded charged simulated reviews, expert adjudication, wages, GPU price, unmeasured monetary fees or future hardware costs are silently set to measured zero.

## One recommended WSL run

From Ubuntu after downloading the ZIP into Windows Downloads, use the terminal block in the accompanying ChatGPT response. In a shell, the main invocation is:

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
"$PY" phase12b_latency_cost.py \
  --repo "$HOME/trustworthy-ai-monitor-fusion" \
  --out-dir "$PRIVATE_DIR/phase12b_latency_20261112" \
  --relative-gpu-cost 1.0 \
  --run-phase11 "$HOME/trustworthy-ai-monitor-fusion/reviewer_analysis/Trustworthy_AI_Phase11_Published_Budget_Benchmark_20261112/phase11_benchmark.py" \
  --db "$PRIVATE_DIR/cases_private.sqlite" \
  --populations 8 --repetitions 25 --budget 4800
```

**Interpretation:** rho=1 is just one normalized cost-sensitivity scenario, not an empirical hourly-price ratio. A substantive economic comparison still needs price quotes or justified labor/compute exchange rates, and an empirical outcome reference to support latent-Y claims.
