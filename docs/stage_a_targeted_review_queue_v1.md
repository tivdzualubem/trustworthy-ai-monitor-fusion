# Stage-A targeted review queue

This step prepares a small queue for explicit primary-category review. It performs no scientific review, base-intent selection, pair generation, semantic-equivalence or severity review, guard scoring, W0, or fresh confirmation. Prof. Bader Rasheed's 2026-09-21 identification-first direction and the existing Stage-A redesign remain controlling. No prospective claim or allocation is frozen.

## Reproduce

Start from the no-decisions candidate snapshot produced by `scripts/select_stage_a_public_base_intents.py` (899 candidates: JailbreakBench 100, WildGuard 749, BeaverTails 50). Then run:

```bash
python scripts/build_stage_a_review_queue.py
python -m unittest discover -s tests -p 'test_stage_a_targeted_review_queue.py'
```

The queue builder uses only the standard library and the local candidate CSV; it makes no network or model calls. It writes `review_queue.csv`, `audit.json`, and `audit.txt` under `results/stage_a_targeted_review_v1/`. The audit records input, configuration, ontology, provenance-manifest, script, and queue hashes. Ordering and output are reproducible for the same inputs. Existing review content is rejected rather than erased.

## Screening rules

The exact source labels and regular expressions are in `configs/stage_a_review_screening_v1.json`. They nominate candidates only. All five review fields, including `independent_intent_group`, stay blank. The input text, text hash, source, revision, and other provenance columns are copied unchanged. The existing hash normalizes whitespace and case; it does not establish semantic independence.

| Category | Screening evidence | Main limitation |
| --- | --- | --- |
| C1 | Physical-harm source labels or violence/weapon phrases | Drug, poisoning, sexual-harm, and self-harm overlaps are not resolved automatically. |
| C2 | Fraud, theft, forgery, scam, or financial-crime phrases; source labels corroborate | Broad illegal-activity labels alone do not nominate. |
| C3 | Named drugs, drug terms, and specific hazardous-substance phrases | Generic chemical/substance mentions are insufficient; no dedicated source-label pool exists in this snapshot. |
| C4 | Cyberattack/Malware-Hacking labels or specific cyber-abuse phrases | Defensive and authorized uses still require exclusion during review. |
| C5 | Self-harm phrases; registered BeaverTails self_harm source hint | BeaverTails can nominate only C5; support/prevention and mixed intents may fail review. |
| C6 | Hate/harassment/discrimination labels or targeted-abuse phrases | The frozen category requires an actual person/group target and harmful content. |
| C7 | Sexual-content labels or sexual-harm phrases | Benign discussion and health topics are not automatically disallowed. |
| C8 | Personal-privacy labels or specific personal-information phrases | Organizational secrecy alone is insufficient; authorization must be reviewed. |

A candidate with multiple category signals is withheld from the main queue, with its ID, hash, and evidence recorded in the audit. This is a screening ambiguity flag, not a final exclusion or finding of mixed intent. BeaverTails is restricted to C5 nomination; the restriction does not establish that its content has no competing primary harm. Reviewers must still reject mixed, ambiguous, benign, defensive, and other-primary-category cases. No candidate appears in more than one category queue.

The queue takes up to 25 singleton nominations per category, prioritizing corroborating label/text evidence, then label-only evidence, then text-only evidence. Within each tier, a SHA-256 ordering with seed 20260921 determines the order. Pools below 20 are reported without padding. Remaining singleton nominations are listed as reserves. This is a purposive screening queue, not a representative sample, a safety estimate, or a guarantee of 13 accepted independent intents.

## C3 probe and review handoff

The original inline C3 probe and its 36 IDs were recovered from the user's `stage_a_c3_c5_gap_probe.txt`. The config preserves those exact patterns and IDs. The audit reruns them against the original JailbreakBench/WildGuard sources and reports their relationship to the new queue. Specific additional drug/hazard phrases extend the nomination pool transparently. Weapons-primary cases belong in C1; drug/hazardous-substance-primary cases may satisfy C3, but mixed cases must be excluded during review. No such judgment is made by this script.

Reviewers must assign an explicit primary category and rationale, and group duplicate/near-duplicate underlying intents across sources. Severity and semantic equivalence are separate later reviews. No row is accepted merely because it was queued.

The existing selector's `--decisions` interface requires all candidate IDs. This 200-row screening queue is **not** a completed decisions file for that interface. A later reviewed-subset handoff must distinguish unreviewed candidates from scientific exclusions; do not fabricate 699 exclusion decisions. This task leaves selection behavior unchanged.
