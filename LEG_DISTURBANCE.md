# Physical disturbance and sham-yoked neural control

This finite experiment asks whether the present full-MaleCNS foreleg loop
changes its recovery from an external disturbance through sensory feedback.
It is a gate toward the whole-fly objective, not a replacement for that goal.

`probe_leg_disturbance.py` uses the original fixed-base FlyMimic foreleg and the
existing all-15-muscle named-pool mapping. No muscle strength, neural gain,
passive mechanics, or motor recruitment parameter is changed. A physical tibia
joint torque of +0.1 or -0.1 **model units** is applied from 0.80 to 0.85 seconds
in a two-second trial. This is an experimental disturbance, not a controller;
its size is not claimed to reproduce a measured biological perturbation.

For each torque sign and each of the two unresolved claw-sensory polarities,
three independent physical/CNS states run together:

- Sham: no torque pulse; senses its own leg.
- Feedback: torque pulse; senses its own leg.
- Yoked sensory: same torque pulse, but senses the matched sham leg.

The yoked condition preserves the entire undisturbed sensory history rather
than freezing at an arbitrary initial angle. Its neural rates and commands
must equal the sham's exactly. Feedback versus yoked differences then isolate
the effect of perturbation-dependent sensory input within this model.
Return toward the sham trajectory alone is not evidence of a neural reflex:
passive mechanics and tonic muscle activation can produce it too.

The primary descriptive metric is absolute tibia-angle deviation from the
matched sham, summed at 10-ms intervals from pulse end through two seconds.
Both signs and polarities are retained, including worsened recovery. No passing
threshold, polarity selection, or physiological validation is inferred from
a small favorable numerical difference.

`verify_leg_disturbance.py` checks identical pre-pulse trajectories, exact yoked
sensory/motor histories, pulse timing and signs, angle-to-current transduction,
and muscle commands. With `--xml`, it also replays 72 physical intervals around
the pulse and later recovery from saved complete integration states. This
checks physical execution, not an independent regeneration of CNS dynamics.

The sensory model still maps angle directly to candidate afferent currents,
without organ strain mechanics or measured tuning. Normalized motor rates are
still averaged without calibrated motor-unit recruitment. A successful numerical
feedback response would not by itself validate either approximation.

## Execution notes

Trial 002 completed twelve lanes for two simulated seconds in 30.88 wall
seconds. Peak tibia displacement from sham was 0.02754–0.02794 radians.
Post-pulse absolute error was 0.000616–0.000634 rad·s in both conditions.

| Sensory polarity | Torque sign | Relative recovery-error reduction from feedback |
| --- | --- | ---: |
| +1 | + | -0.000977% |
| +1 | - | +0.002772% |
| -1 | + | +0.000902% |
| -1 | - | -0.001135% |

Positive values mean less error than the yoked control; negative values mean
more. These tiny, mixed-sign differences do not demonstrate useful correction.
Feedback versus yoked tibia differences never exceed 1.064e-6 radians, despite
the approximately 0.028-radian disturbance. Control differences are at most
7.60e-6 normalized activation. Apparent recovery is therefore overwhelmingly
present without disturbance-dependent neural feedback in this assay.

All control reconstruction and 72 physical interval replays matched exactly.
The evidence is in `evidence/trials/leg-disturbance-002/` and
`evidence/leg-disturbance-verification-002.json`. The default local test suite
also passed 21 tests with one dependency-related skip; it does not validate
the new biological interpretation. This experiment ended; no service was added.

The next intervention should address the uncalibrated sensorimotor transfer,
using physiological response and recruitment evidence, rather than run this
same system longer or promote its mechanically produced recovery as a reflex.
The [calibration source index](REFLEX_CALIBRATION.md) registers physiological
records and exclusions while keeping unavailable measurements and unresolved
motor-cell identities explicit.

The first launch, `leg-disturbance-001`, stopped before simulation because the
mapping file was not present at the requested remote path. After transferring
the existing mapping unchanged, the experiment used a fresh output directory,
`leg-disturbance-002`. The incomplete directory was not overwritten.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python probe_leg_disturbance.py \
  --source ../chreatures --model ../releases/chreatures-v5-garden/model \
  --mapping evidence/pooled-muscle-mapping-001.json \
  --sensory proprioceptor-annotations.json \
  --xml ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --output runs/leg-disturbance
.venv/bin/python verify_leg_disturbance.py --trial runs/leg-disturbance \
  --mapping evidence/pooled-muscle-mapping-001.json \
  --xml ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --output runs/leg-disturbance-verification.json
```
