# Stage-A Theory Notes

## Marginal FNR under changing composition

Let `e` index an environment or formulation condition and let `Z` denote case characteristics such as harm category, severity, and base intent. For a fixed guard,

\[
\mathrm{FNR}_e
=
P(D=0\mid Y=1,e)
=
\sum_z
P(D=0\mid Y=1,Z=z,e)
P(Z=z\mid Y=1,e).
\]

A marginal FNR difference across environments can therefore arise from either a change in conditional guard behaviour or a change in the composition of harmful cases. Without fixing, identifying, or standardizing the composition distribution, the marginal difference alone does not identify guard degradation.

The Stage-A paired design addresses this by evaluating the same underlying base intent across controlled formulation changes. Strict source comparisons additionally require semantic retention and severity alignment.

## Paired versus independent information requirements

For paired decisions define

\[
X_i=D_{i,R}-D_{i,L}, \qquad X_i\in\{-1,0,1\}.
\]

With

\[
q_{10}=P(D_L=1,D_R=0), \qquad
q_{01}=P(D_L=0,D_R=1),
\]

and

\[
q=q_{10}+q_{01},
\]

we have

\[
E[X]=q_{01}-q_{10}
\]

and

\[
\operatorname{Var}(X)
=
q-(q_{01}-q_{10})^2.
\]

The variance of a paired decision difference is therefore governed primarily by the discordance probability `q`. When `q` is small, most paired observations cancel exactly and pairing can be substantially more efficient than estimating two independent marginal proportions. As `q` increases, that advantage diminishes.

Stage-A uses `q` as an empirical design diagnostic rather than as a safety metric. The observed values indicate potentially useful pairing for some direct-source comparisons but not uniformly after representation change.

## Common-mode reliability

Let `M_g=1` denote a miss by guard `g`. Multi-guard reliability depends on the joint distribution

\[
P(M_1,\ldots,M_G),
\]

not only on the individual marginal FNRs. In particular,

\[
P(M_1=\cdots=M_G=1)
\]

cannot be recovered from marginal miss rates without dependence assumptions.

Stage-A therefore reports joint miss counts on the same underlying cases. The obfuscated conditions show substantial three-of-four co-miss concentration, particularly across ShieldGemma, Granite Guardian, and Qwen3Guard, while no four-of-four miss was observed.

This supports further investigation of common-mode vulnerability. It does not justify a parametric common-cause model or a deployment-wide reliability claim from Stage-A alone.
