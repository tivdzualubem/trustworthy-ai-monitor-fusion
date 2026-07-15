# Final v3 policy timing package

This repeats the end-to-end policy benchmark using the author-reviewed v3
fusion bundle and locked target-FPR 0.05 route.

Protocol:

- 128 deterministic audited-label calibration examples
- Tesla T4
- batch size 1
- eight excluded warm-up rows per complete policy
- alternating full/selective execution order
- CUDA synchronization
- official Qwen3Guard chat-template generation and structured parsing
- pinned compact and Qwen model revisions
- no model refitting

On a Kaggle Tesla-T4 notebook, extract this package, set `HF_TOKEN`, then run:

```bash
export HF_TOKEN='your-read-token'
bash /path/to/final_v3_policy_timing_package/run_gpu.sh \
  /path/to/final_v3_policy_timing_package \
  /kaggle/working/final_v3_policy_timing_results
```

Download `/kaggle/working/final_v3_policy_timing_results.zip`.

The runner records regenerated-versus-cached Qwen classification differences in `qwen_score_mismatches.csv`. These differences are measurement-reproducibility evidence and do not discard the latency run.
