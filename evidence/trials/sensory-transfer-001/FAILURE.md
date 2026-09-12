# Failed report, retained trace

The first invocation completed 400 neural steps and saved `trace.npz`, then
failed while constructing its JSON report:

```text
ValueError: operands could not be broadcast together with shapes (400,815) (400,4)
```

The comparison selected `trace['motor_rates'][:, 0]` instead of
`trace['motor_rates'][:, :, 0]` from the source embodied trace. Its axes are
time, neuron, resident. No model or inference change was needed. The corrected
runner checks the source comparison shape before inference; the complete rerun
is `sensory-transfer-002`. This directory has no success receipt. This note is
a retrospective record of the observed terminal error, not captured stderr.
