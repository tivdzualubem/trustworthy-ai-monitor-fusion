# Budget-Aware Runtime Safety Monitor Fusion

This repository contains the implementation and evaluation of a budget-aware
multi-monitor policy for runtime AI safety.

## Primary research question

Can a simple budget-aware multi-monitor policy improve the
recall-cost-false-positive frontier over strong baselines, while providing
valid in-distribution calibration bounds and revealing how those bounds fail
under held-out attack-family shift?

## Prediction unit

Each sample is a prompt-response pair:

X = (prompt, response)

## Primary target

Y = 1 when the response should be intercepted because, in the context of the
prompt, it materially provides or contains disallowed harmful content.

Y = 0 for benign responses, safe refusals, and safe redirections.
