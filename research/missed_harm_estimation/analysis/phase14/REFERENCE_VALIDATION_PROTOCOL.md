# Reference-outcome validity — concrete study protocol and stop/go rules

**Not a new Phase 9D R2-focused study.** This protocol addresses the measurement target, not a single reviewer. The archived 96-case human study has 336 judgments but no independently certified Y. It is a pilot for review-cost and disagreement analysis, not a latent-harm prevalence validation set.

## A. State the estimand before collecting any new labels

- **Historical finite-frame estimand:** `A_Y = N^{-1} sum_{i=1}^N 1{D_i=0} Y_i` for the frozen 1,687 historical prompt–response records and the recorded historical monitor decision D. This is not a prospective-deployment estimate. Historical benchmark `historical_y` is not reference truth.
- **Prospective target estimand (separate):** exactly the same expression on a newly defined, date-delimited, version-frozen stream of target assistant responses with native D. Its frame, sampling and response generation must be independently recorded. Do not silently transport results from historical data.
- **Observable review-protocol estimand:** `A_Z` is defined only after freezing a reproducible qualified-reviewer/adjudication protocol for response-level C1–C8 outcomes, including how Uncertain is handled. Estimating A_Z does not, without justified reviewer error assumptions, identify A_Y.

## B. Historical study forensic check, without reopening the R2 branch

1. Preserve raw 96 case choices, full 6 source × historical D strata, the exact archived counts `N_h` and sampled `n_h`, reviewer instructions and case/monitor version hashes privately. Audit how each stratum was sampled; do **not** assert positive known individual inclusion probabilities unless the original selection mechanism verifies them.
2. Preserve R1–R4 independently. R4 was selected based on primary votes with a nine-case audit; R4 is not a random gold-standard reviewer. No majority or historical benchmark labels are automatically true Y.
3. Produce aggregate-only descriptive disagreement and time reporting. Do not export case IDs, text, reviewer credentials or free-text rationales.
4. If original inclusion probabilities are not recoverable, treat design-weighted numbers as exploratory and do not label them design-unbiased estimates.

## C. Prospective confirmatory study — minimum defensible sequence

1. Freeze the population source, target assistant version, decoding settings, target response policy/C1–C8, native monitor versions, threshold and decision D, time window, dedup/dependence policy, and all inputs/outputs within a **private** collection manifest. Record exact N and allowed M. The deployment population must be clearly distinguished from the historical development records.
2. Precommit sampling: **simple random sampling without replacement from all M allowed responses**, using a recorded random seed. Every allowed case has inclusion probability `pi=n/M`, with known frame and distinct cases. Source-stratified variants require separately known `n_h`, `M_h` and proper variance formulas; the included exact hypergeometric interval applies ONLY to unstratified SRSWOR of the allowed frame.
3. Blind qualified human reference reviewers to the target monitor decision, historic labels, model scores, proposal and one another's answers. Provide the frozen policy page and examples that distinguish disallowed *assistant-response* assistance from merely unsafe user requests and safe refusals. Collect Safe/Harmful/Uncertain plus response-level category rationale privately. Avoid reusing untrained project organizers as an unqualified gold standard.
4. Predefine independent disagreement resolution (e.g., blinded qualified adjudicator), conflicts-of-interest tracking, category mismatch handling, and an independent quality-audit subset chosen independently of disagreements. Record all review, training and adjudication time. Ambiguity is never silently mapped to Safe or discarded.
5. **Even qualified consensus is not mathematically perfect latent Y**: call its result an operational reference endpoint Z*, quantify unverifiable accuracy limits, and state what external accuracy evidence/assumptions would justify inference about latent Y.
6. Independently freeze the data *before* choosing estimation algorithms. Disallow calibration/selection on the confirmatory holdout. Record probability inclusion and any nonresponse; positive inclusion probabilities do not survive selective missing review without appropriate handling.
7. Analyze A_Z*: with n verified *binary* Z* labels and x Harmful, sample the allowed frame by SRSWOR and invert the finite-population hypergeometric distribution. Code `phase14_finite_population_interval.py` produces a conservative exact 95% interval for `A_Z* = K/N`. If u cases remain Uncertain, return worst-case lower/upper by treating them as Safe for the lower bound and Harmful for the upper. Neither is an error-corrected interval for A_Y.
8. Preserve actual review-time seconds and measured hardware latency separately; only a declared economic conversion (with GPU costs and wages) can turn these into a common cost unit. Include calibration, proxy calls, adjudication and training fees; don't charge historical acquisition zero without disclosure.

## D. Stop/go before final research review paper claims

- **GO for a historical reviewer-endpoint pilot report:** immutable data scope; reproducible sampling weights; explicit Z definition and policy; design-based uncertainty; review cost and limited claims.
- **GO for latent-Y safety/risk claim:** independent externally justified reference validity or stated sensitivity bounds, calibrated sampling; native deployment frame, monitor/response provenance and source-transport restrictions.
- **NO-GO for a claimed official MultiPPI/AM-PPI failure:** using restricted adapters, synthetic proxies and unvalidated labels instead of author-equivalent methods, native predictor measurements and checked inference.
- **NO-GO for a guaranteed realized monetary budget:** only model latency + assumed GPU-to-human price conversion is currently available.
- **NO-GO for publication-ready novelty:** generic nonidentification, interval clipping, elementary Bernoulli HT correction, hypergeometric inversion and convex first-order gap certificates are established mathematics, not new research contributions.

A real confirmation study cannot be completed from these uploaded aggregate JSON files; it requires genuine responses, frozen monitor decisions and independent human judgments. This protocol is implementable without resurrecting the 12-case R2-only review.
