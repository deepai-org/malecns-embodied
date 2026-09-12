# Whole-body torque and approximate antagonist muscles

The original host's zero joint command means a powered position target. The
experimental host offers `advance_torque`, a separate protocol operation using
the same 92-float packet shape. Channels 0–83 instead specify signed fractions
of the original actuator force capacity. Channels 84–91 must be zero, disabling
adhesion and the oral actions. It advertises
`experimental_torque_mode: normalized-force-clamp-v1`; do not send torque
packets to an unpatched host or confuse these semantics with the original
actuator schema. Original `advance` retains its original meaning.

The implementation collapses each enabled joint actuator's force range onto
the requested force. This cancels the position-servo contribution, including
its affine bias. Requests remain bounded by the host's current force capacity.
This is an experimental force interface, not a new anatomical muscle asset.
The existing ecology capacity/work model was designed for servo commands and
is **not validated for torque-driven muscle metabolism**. Snapshot/replay and
long-term ecological compatibility of the new mode have not been verified.

## Mechanical contract test

`probe_torque_host.py` ran three fresh one-second worlds, each with four bodies:

| Command | Maximum actuator force, model units |
| --- | ---: |
| All zero, torque mode | exactly 0 |
| One tibia pulse, torque mode | 0.249789 |
| All zero, original position mode | 3.007776 |

Every non-target actuator stayed exactly unpowered during the pulse; all
actuator forces returned to zero after it. The host rejected a nonzero
adhesion channel without advancing physical time. Source, scene and binary
hashes and numeric traces are in `evidence/trials/torque-host-001`.

## Approximate muscles, no prescribed gait

`AggregateLegMuscles` groups annotated Tr flexor/extensor and Ti flexor/extensor
motor neurons by leg and side. These drive 24 aggregate contractile units
across twelve hinges in each whole body. This is a coarse functional grouping,
not identified muscle geometry; accessory muscles and 72 other joint actuator
channels remain unassigned and unpowered. No learned motor readout is used.

Activation is a first-order filter of the cohort's mean normalized motor rate,
with assumed 20-ms time constant. For activations `f,e`, the normalized torque is:

```
0.03 * polarity * (f-e)
  + 0.03 * (f+e) * (fixed_rest_angle - joint_angle)
  - 0.001 * (f+e) * joint_velocity
```

Torque is clipped to [-1,1] and held for a 10-ms control interval. The
contractile/elastic structure is inspired by NeuroMechFly's
[spring-damper muscle model](https://github.com/NeLy-EPFL/NeuroMechFly/blob/main/NeuroMechFly/control/spring_damper_muscles.py),
but its oscillator-based activation is not used. All gains here are uncalibrated
hypotheses, not fitted or experimentally measured physiology. The rest angle
comes from the pinned body's rig pose. Consequently co-contraction favors
that pose by construction; this must not be mistaken for neural postural skill.
Both signs for flexor/extensor coordinate polarity were run and retained;
neither is selected for desirable motion. Exact moment-arm geometry is missing.

Physical retinal and body samples enter the original full-CNS sensory adapter.
Its broad, initialized mappings remain approximate. The CNS motor decoder is
computed by the existing forward API but its output is discarded; only the
815 raw motor-neuron rates can enter the named muscle interface. There is no
trajectory target, body-attitude controller, oscillator, reward or external
context input to the muscles.

## Two-second whole-body trials

All four trials completed with finite recorded state. Each contains four
residents sharing a scene, not four independent replicates.

| Trial | Max thorax difference from intact (mm) |
| --- | ---: |
| `aggregate-body-zero-001`, motor input zero from initialization | 0.0061343 |
| `aggregate-body-frozen-001`, retinal/body inputs frozen at initial sample | 0.0034431 |
| `aggregate-body-reverse-001`, reverse candidate contractile sign | 0.0039006 |

The intact trial is `aggregate-body-001`. In its final second, each body's
horizontal path was only 0.0064–0.0117 mm. All conditions largely settled.
Upright bodies and these small path lengths are not useful locomotion.

`verify_aggregate_body.py` checked initial-state matching, physical timestep,
finite traces, runner/muscle/annotation hashes, every activation update and
muscle command, and exact zero force on unsupported actuators after stepping.
Recorded commands recomputed exactly in all four trials. The zero-motor trial
also has exactly zero actuator force everywhere after stepping. The ten local
unit tests and four new tests on the GPU host passed.

The same audit found **504 non-free joints, all with stiffness 10 and damping
0.5 in the original model units**, including neutral-angle spring references
in the legs. These remain active with zero actuator force. Removing the servos
therefore does not prove the upright posture comes from the CNS. This is a
specific next mechanical question to test, not a reason to tune a gait until it
looks convincing. The full sensory-organ model, broad muscle coverage and
rich behavior are still missing.

## Reproduce without changing the original baseline

Create a separate checkout at the pinned Chreatures commit from `sources.json`.
Apply the public patch there and build with the MuJoCo/Rust instructions in
README.md. For example, from this experiment repository:

```sh
git -C ../chreatures-torque apply --check "$PWD/patches/experimental-torque.patch"
git -C ../chreatures-torque apply "$PWD/patches/experimental-torque.patch"
# Build ../chreatures-torque/native/fly-world with the pinned MuJoCo 3.12.0.
.venv/bin/python probe_torque_host.py --source ../chreatures-torque \
  --scene ../chreatures-torque/native/fly-body/scenes/training-4/world.json \
  --output runs/torque-host
.venv/bin/python run_aggregate_body.py --source ../chreatures-torque \
  --model ../releases/chreatures-v5-garden/model \
  --scene ../chreatures-torque/native/fly-body/scenes/training-4/world.json \
  --motor motor-annotations.json --output runs/aggregate-body --steps 200
```

Use `--condition zero-motor`, `--condition frozen-sensory` or `--polarity -1`
with fresh output paths for the other trials. The published trace comparison
can be reproduced using the pinned scene and bundled trial evidence:

```sh
OPENBLAS_NUM_THREADS=1 python3 verify_aggregate_body.py \
  --scene ../chreatures/native/fly-body/scenes/training-4/world.json \
  --output runs/aggregate-verification.json
```

These are finite experiments. No unattended service or new behavioral policy
has been installed. The original host and source remain unchanged.
