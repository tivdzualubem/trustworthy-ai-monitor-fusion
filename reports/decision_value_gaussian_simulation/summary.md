# Two-context Gaussian decision-value simulation

## Scope

This is a controlled toy experiment. It does not establish a new routing
method and does not use the real-data test partition.

## Data-generating process

- Equal class prior.
- Context A probability: 0.50.
- Cheap signal: H given Y is Gaussian with class means plus or minus
  0.80 and unit variance.
- The cheap-signal distribution is identical across contexts.
- In context A, the optional monitor is conditionally independent of H given
  Y and has Gaussian class separation
  2.00.
- In context B, the optional monitor is H plus independent Gaussian noise and
  is conditionally redundant given H.
- Train examples per seed: 6000.
- Test examples per seed: 12000.
- Independent seeds: 30.

## Policies

All policies acquire exactly the same number of optional-monitor outputs at
each requested acquisition rate.

1. Random routing.
2. Cheap-model uncertainty routing.
3. Oracle decision-value routing using the analytic conditional Bayes-risk
   reduction.
4. A learned value estimator trained on realized decision-loss reduction
   using only pre-acquisition H and the observed context.

## Primary matched-budget results

At 25% acquisition:

- uncertainty decision loss:
  0.161797;
- learned-value decision loss:
  0.133958;
- oracle decision-value loss:
  0.132353;
- learned reduction relative to uncertainty:
  0.027839;
- 95% paired interval:
  [0.027368,
   0.028310].

At 50% acquisition:

- uncertainty decision loss:
  0.132450;
- learned-value decision loss:
  0.113794;
- oracle decision-value loss:
  0.113628.

## Diagnostic checks

Across seeds:

- mean context-A cheap decision loss:
  0.208108;
- mean context-A full-information loss:
  0.015381;
- mean context-B cheap decision loss:
  0.211682;
- mean context-B full-information loss:
  0.211682;
- mean context-A cheap uncertainty:
  0.211742;
- mean context-B cheap uncertainty:
  0.212154;
- learned-value Spearman correlation with analytic value:
  0.939564.

The cheap uncertainty distribution is matched across contexts, but the
optional monitor has positive decision value only in context A. The learned
value estimator uses the context to allocate acquisition to informative
examples and improves decision loss relative to uncertainty routing at the
same acquisition rate.

## Interpretation

The experiment demonstrates the mechanism requested by the professor:
ordinary uncertainty can be identical across examples while optional-monitor
quality differs. A value estimator can exploit legitimate pre-acquisition
context to improve matched-budget routing.

This is only a toy validation. The next step is the development-only,
cross-fitted real-data decision-value diagnostic.
