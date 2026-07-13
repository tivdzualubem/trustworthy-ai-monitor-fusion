# Final audited-label dataset

The final author-reviewed audit decisions were applied to the unified dataset.

## Counts

- Total examples: 2159
- Author-reviewed examples: 200
- Corrected labels: 19
- Original class counts: {0: 1765, 1: 394}
- Audited class counts: {0: 1756, 1: 403}

## Corrected-label Qwen metrics

- prompt_only: ROC-AUC=0.8308, AP=0.4236, recall=0.8908, FPR=0.2597, precision=0.4405
- response_only: ROC-AUC=0.9055, AP=0.7613, recall=0.6873, FPR=0.0188, precision=0.8935
- prompt_response: ROC-AUC=0.9256, AP=0.7677, recall=0.8139, FPR=0.0347, precision=0.8432

The monitor scores were not regenerated because label correction does not
change monitor inference outputs. The score cache was rebuilt with the
final audited target labels and audit provenance.
