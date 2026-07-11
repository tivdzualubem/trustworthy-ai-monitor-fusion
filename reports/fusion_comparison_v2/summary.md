# Serialized full fusion versus selective acquisition

Both policies load the exact serialized fitted bundle. No model is
reconstructed or refit during evaluation.

Operating points were selected only on `policy_selection` and then locked
before evaluation on calibration, final-test, and held-out-shift splits.

The expensive-monitor comparison is reported as call rate and call reduction.
Latency claims are intentionally deferred until a controlled synchronized
timing benchmark is completed.

These are provisional development results because the assistant-assisted
label audit still requires author review. They are not a formal 5% FPR
risk certificate.

## Evaluation

- calibration, target=0.010, full_information_always_on: recall=0.1389, FPR=0.0029, precision=0.9091, expensive-call-rate=1.0000
- calibration, target=0.010, selective_acquisition: recall=0.0972, FPR=0.0229, precision=0.4667, expensive-call-rate=0.2559
- calibration, target=0.025, full_information_always_on: recall=0.4306, FPR=0.0114, precision=0.8857, expensive-call-rate=1.0000
- calibration, target=0.025, selective_acquisition: recall=0.4306, FPR=0.0314, precision=0.7381, expensive-call-rate=0.3649
- calibration, target=0.050, full_information_always_on: recall=0.7083, FPR=0.0257, precision=0.8500, expensive-call-rate=1.0000
- calibration, target=0.050, selective_acquisition: recall=0.7083, FPR=0.0400, precision=0.7846, expensive-call-rate=0.7583
- calibration, target=0.100, full_information_always_on: recall=0.8750, FPR=0.1257, precision=0.5888, expensive-call-rate=1.0000
- calibration, target=0.100, selective_acquisition: recall=0.8750, FPR=0.1314, precision=0.5780, expensive-call-rate=0.8294
- final_test, target=0.010, full_information_always_on: recall=0.1806, FPR=0.0000, precision=1.0000, expensive-call-rate=1.0000
- final_test, target=0.010, selective_acquisition: recall=0.1389, FPR=0.0114, precision=0.7143, expensive-call-rate=0.2085
- final_test, target=0.025, full_information_always_on: recall=0.4583, FPR=0.0200, precision=0.8250, expensive-call-rate=1.0000
- final_test, target=0.025, selective_acquisition: recall=0.4861, FPR=0.0314, precision=0.7609, expensive-call-rate=0.3460
- final_test, target=0.050, full_information_always_on: recall=0.7500, FPR=0.0343, precision=0.8182, expensive-call-rate=1.0000
- final_test, target=0.050, selective_acquisition: recall=0.7500, FPR=0.0457, precision=0.7714, expensive-call-rate=0.7417
- final_test, target=0.100, full_information_always_on: recall=0.8750, FPR=0.0971, precision=0.6495, expensive-call-rate=1.0000
- final_test, target=0.100, selective_acquisition: recall=0.8750, FPR=0.1029, precision=0.6364, expensive-call-rate=0.8555
- held_out_shift, target=0.010, full_information_always_on: recall=0.1471, FPR=0.0000, precision=1.0000, expensive-call-rate=1.0000
- held_out_shift, target=0.010, selective_acquisition: recall=0.0294, FPR=0.0625, precision=0.5000, expensive-call-rate=0.5000
- held_out_shift, target=0.025, full_information_always_on: recall=0.6176, FPR=0.1875, precision=0.8750, expensive-call-rate=1.0000
- held_out_shift, target=0.025, selective_acquisition: recall=0.5882, FPR=0.2500, precision=0.8333, expensive-call-rate=0.5600
- held_out_shift, target=0.050, full_information_always_on: recall=0.9118, FPR=0.3750, precision=0.8378, expensive-call-rate=1.0000
- held_out_shift, target=0.050, selective_acquisition: recall=0.8824, FPR=0.4375, precision=0.8108, expensive-call-rate=0.8400
- held_out_shift, target=0.100, full_information_always_on: recall=1.0000, FPR=0.6250, precision=0.7727, expensive-call-rate=1.0000
- held_out_shift, target=0.100, selective_acquisition: recall=1.0000, FPR=0.6250, precision=0.7727, expensive-call-rate=0.7800

## Paired comparison

Differences are selective minus full-information. Confidence intervals
use 2000 stratified paired bootstrap replicates.

- calibration, target=0.010: Δrecall=-0.0417 [-0.0972, 0.0139], ΔFPR=0.0200 [0.0086, 0.0343], expensive-call reduction=0.7441 [0.7037, 0.7844]
- calibration, target=0.025: Δrecall=0.0000 [-0.0417, 0.0417], ΔFPR=0.0200 [0.0086, 0.0371], expensive-call reduction=0.6351 [0.5900, 0.6801]
- calibration, target=0.050: Δrecall=0.0000 [-0.0417, 0.0417], ΔFPR=0.0143 [0.0000, 0.0314], expensive-call reduction=0.2417 [0.2038, 0.2844]
- calibration, target=0.100: Δrecall=0.0000 [0.0000, 0.0000], ΔFPR=0.0057 [0.0000, 0.0143], expensive-call reduction=0.1706 [0.1351, 0.2085]
- final_test, target=0.010: Δrecall=-0.0417 [-0.1250, 0.0417], ΔFPR=0.0114 [0.0029, 0.0229], expensive-call reduction=0.7915 [0.7512, 0.8294]
- final_test, target=0.025: Δrecall=0.0278 [-0.0278, 0.0833], ΔFPR=0.0114 [0.0029, 0.0229], expensive-call reduction=0.6540 [0.6090, 0.6991]
- final_test, target=0.050: Δrecall=0.0000 [-0.0694, 0.0694], ΔFPR=0.0114 [0.0029, 0.0229], expensive-call reduction=0.2583 [0.2156, 0.3009]
- final_test, target=0.100: Δrecall=0.0000 [0.0000, 0.0000], ΔFPR=0.0057 [0.0000, 0.0143], expensive-call reduction=0.1445 [0.1114, 0.1801]
- held_out_shift, target=0.010: Δrecall=-0.1176 [-0.2353, -0.0294], ΔFPR=0.0625 [0.0000, 0.1875], expensive-call reduction=0.5000 [0.3600, 0.6400]
- held_out_shift, target=0.025: Δrecall=-0.0294 [-0.0882, 0.0000], ΔFPR=0.0625 [0.0000, 0.1875], expensive-call reduction=0.4400 [0.3000, 0.5800]
- held_out_shift, target=0.050: Δrecall=-0.0294 [-0.0882, 0.0000], ΔFPR=0.0625 [0.0000, 0.1875], expensive-call reduction=0.1600 [0.0600, 0.2600]
- held_out_shift, target=0.100: Δrecall=0.0000 [0.0000, 0.0000], ΔFPR=0.0000 [0.0000, 0.0000], expensive-call reduction=0.2200 [0.1000, 0.3400]