# Exact-Cost, Risk-Constrained Cascade Protocol v2

This document is the implementation protocol for the professor-feedback v2
analysis. It is not a manuscript revision. The machine-readable source of truth
is `configs/exact_cost_risk_cascade_protocol_v2.json`.

## Scope and claim boundary

The guaranteed contribution is an evaluation and measurement study of
exact-cost, risk-constrained safety-monitor cascades. A router-superiority claim
is conditional on every fresh confirmatory gate passing. Otherwise, the result
remains an evaluation-methodology paper explaining why apparent routing gains
can disappear under correct comparison.

The prior no-go result is not reversed by this protocol. The old common-risk
frontier is invalid for superiority claims because it did not compare baselines
at exact cost, did not define one jointly feasible random policy, used an
empirical FPR screen without selection-valid joint control, and relied on
test-fold ranks rather than a deployable online threshold.

## Data boundary

The existing `final_test` and `held_out_shift` partitions must not be opened
again. They were exposed in an earlier routing study and are therefore not fresh
confirmatory data. The three old development partitions are treated as one
exploratory development pool.

Implementation begins with synthetic tests. Development work may use only a
development-only view. The two mixed-split parquet containers may be accessed
only by the dedicated predicate-filtered materializer; protected rows may not
be materialized, counted, summarized, logged, hashed, or evaluated. Paper files
remain unchanged until implementation and evidence are complete.

## Policy-specific signed value

For fixed base and augmented downstream policies,

\[
V_\pi=L(Y,\pi_{\mathrm{base}}(H))
      -L(Y,\pi_{\mathrm{augmented}}(H,Z)).
\]

The conditional target is \(v_\pi(h)=\mathbb{E}[V_\pi\mid H=h]\). It is
policy-specific and may be negative. Under heterogeneous incremental cost, the
online policy acquires when

\[
\widehat v_\pi(h)-\lambda\widehat c(h)>0.
\]

For positive costs, the implemented score is
\(\widehat v_\pi(h)/\max(\widehat c(h),1\text{ ms})\). Neither optional-monitor
outputs nor post-acquisition latency may enter the router.

## Required comparisons

The proposed cost-aware signed-value policy is compared with:

1. distance to the frozen base decision threshold;
2. a model predicting whether the current base decision is wrong;
3. deterministic seeded random acquisition;
4. direct full-information fusion as the full-cost endpoint;
5. never-acquire and always-acquire endpoints.

The repeated development analysis uses five grouped-fold seeds and three model
families for signed-value estimation, current-error prediction, and direct
fusion. Hyperparameters are selected only inside the current inner training
partition. Model-family selection uses the frozen across-seed rule.

## Exact cost and online deployment

The primary cost is measured per example from request start through the final
decision. It includes preprocessing, cheap monitors, router features, router
inference, optional monitoring, and final fusion. Incremental cost relative to
the base policy is secondary. A fixed timeout bounds the cost risk, and timeouts
receive a prespecified conservative action.

Every selective policy receives the same absolute millisecond targets. On fresh
calibration optimization data, two adjacent streaming thresholds may be mixed
with a deterministic hash-based probability to match each target in expected
cost. The resulting thresholds and mixture probability are fixed constants at
inference. Evaluation-set ranks are never used.

Recall comparisons are inferential only when paired total costs pass the frozen
equivalence test. A cheaper grid point may not replace an exact-cost comparator.

## Joint FPR and cost control

Fresh calibration is group-split into a 30% optimization subset and a 70% risk
testing subset. Optimization constructs and Pareto-filters at most 100 candidate
policies. The independent risk subset applies Learn-Then-Test jointly to:

- FPR at most 0.05;
- mean total end-to-end latency at most the absolute cost budget.

The candidate p-value is the maximum of the valid one-sided p-values for the two
constraints. Bonferroni correction across the Pareto-filtered candidates controls
family-wise error at 0.05. A point estimate alone cannot certify a policy.

## Fresh confirmation

Fresh calibration and confirmatory sources must not overlap the legacy data,
and confirmatory sources must not overlap fresh calibration. Confirmatory items
must be collected after the protocol and online policy are frozen. Exact and
semantic deduplication occurs before labels or monitor outputs are inspected.

At least three independent raters label every item while blinded to monitor and
policy outputs. Every disagreement is adjudicated. Sample sizes are calculated
before collection from the prespecified FPR and paired-recall power targets.

The sealed confirmatory evaluation runs once. A router-superiority claim requires
all joint FPR, exact-cost, multiplicity-adjusted recall, end-to-end latency, tail
latency, labeling, freshness, and data-boundary gates to pass. Any failure retains
the no-go and the evaluation-methodology framing.
