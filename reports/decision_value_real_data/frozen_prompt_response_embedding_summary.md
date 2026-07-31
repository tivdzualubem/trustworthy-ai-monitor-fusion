# Complete-Text Frozen Prompt-Response Embeddings

This artifact contains frozen embeddings for the 1,687 development examples
only. It excludes `final_test`, `held_out_shift`, labels, source-dataset,
attack-family, group-identifier, and raw-text fields.

## Method

The exact input template is:

```text
[PROMPT]
{prompt}
[RESPONSE]
{response}
```

The complete text is tokenized without truncation. Content tokens are split
into deterministic, non-overlapping chunks. Raw chunk embeddings are combined
using content-token-count weights, and the final 384-dimensional example
vector is L2-normalized.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Resolved revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Transformers: `4.57.6`
- Model sequence length: 256
- Maximum content tokens/chunk:
  254
- Total chunks: 3609
- Multi-chunk examples: 1094
- Maximum chunks/example: 10
- Total content tokens: 699243
- Covered content tokens:
  699243
- Minimum token coverage:
  1.000000
- Truncated examples: 0
- End-to-end cost:
  159.468164 ms/example

This development-only artifact does not itself establish value predictability
and does not itself pass the professor's milestone.
