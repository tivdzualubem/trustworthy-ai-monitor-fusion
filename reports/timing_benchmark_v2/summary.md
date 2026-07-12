# Controlled timing benchmark v2

## Protocol

- Hardware: Tesla T4
- Benchmark rows: 128
- Split: `calibration`
- Batch size: 1
- Warm-up excluded: True
- CUDA synchronization: True
- Models refitted during timing: False

## Monitor latency

- Rule filter p50: 1.04 ms
- Compact monitor p50: 38.09 ms
- Qwen prompt-only p50: 641.50 ms
- Qwen response-only p50: 801.90 ms
- Qwen prompt-response p50: 1368.64 ms

## End-to-end policy timing

- Full-information mean: 1528.84 ms
- Selective-acquisition mean: 1268.87 ms
- Mean reduction: 259.98 ms (17.0%)
- Full-information p50: 1449.02 ms
- Selective-acquisition p50: 1321.90 ms
- p50 reduction: 127.12 ms (8.8%)
- Selective expensive-monitor call rate: 78.1%
- p95 difference, selective minus full: 8.34 ms
- p99 difference, selective minus full: -0.08 ms

Selective acquisition reduces mean and median latency because it avoids Qwen
for 21.9% of examples. It does not improve tail latency: p95 and p99 remain
approximately equal because slow examples are generally routed through the
expensive monitor.

These timing savings do not reverse the provisional no-go routing decision,
because the held-out-shift safety constraint was not preserved.
