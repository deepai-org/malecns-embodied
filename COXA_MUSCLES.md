# Adding the missing leg-swing coordinate

The earlier twelve-hinge actuator covered trochanter and tibia bending but left
all coxa coordinates unpowered. `CoxaLegMuscles` adds six fixed coxa coordinates,
one per leg, driven by 34 named sternal anterior/posterior rotator motor neurons.
The full loop now has eighteen actuated leg hinges and 36 aggregate antagonist
units. Twenty-four of the rig's 42 leg actuator channels remain unpowered.

The anatomical basis is that these named motor groups oppose anterior and
posterior coxa rotation. This is described in the
[MANC motor-circuit study](https://elifesciences.org/articles/96084), especially
the leg premotor and DNp02/DNp11 pathways. Motor IDs and exact cohorts come from
the authenticated MaleCNS annotations, not behavioral labels or a trained readout.

## Fixed coordinate approximation

We do **not** have measured muscle insertion geometry for these six rig legs.
`map_coxa_protraction.py` therefore makes a limited, explicit approximation:
at the original rig keyframe it perturbs each coxa coordinate by ±0.0001 rad,
measures knee-origin displacement along the thorax-to-head direction, and selects
the coordinate with the largest absolute derivative. This is done once, offline.

For this rig, all six selected coordinates are labeled `roll`, with negative
rotation producing anterior displacement locally. The derivative magnitudes
are approximately 0.238, 0.507 and 0.508 mm/rad for front, middle and hind legs.
The matching left/right results provide a symmetry check, not biological validation.
The complete candidates and source hashes are in `evidence/coxa-map-001.json`.

The runtime receives the fixed mapping. It does not optimize the axis, inspect
body heading, set a desired foot position, or generate a swing trajectory. The
same approximate activation filter and force law as the existing twelve hinges
are reused. Motor cohorts are swapped when required by the fixed coordinate sign.
Rates remain normalized model values; no spike-to-force calibration is claimed.

This is not a three-dimensional anatomical muscle model. Local coordinate sign
and mechanical advantage can change away from rest. Other coxa muscles, femur
reduction, distal leg muscles and their coupled mechanics remain incomplete.
More actuation does not automatically mean better behavior.

## Whole-body test

The original code, model weights and earlier trials remain unchanged. The new
`run_coxa_muscles.py` uses the physics-step muscle host, named claw sensory
adapter and no-joint-passives scene. Finite driven, frozen-sensory and zero-muscle
trials are stored under `evidence/trials/coxa-*-001`. Each world contains four
residents; these are not independent replicates. Frozen-sensory holds all inputs,
not only the named claw rows. The previous named-senses driven trial is the
no-coxa-actuation comparison, with the same seed and unchanged existing gains.

The initial runs still tip/settle. No useful locomotion or rich behavior is
claimed. Completing an efferent pathway is an implementation improvement, not
evidence of a better behavioral result. Do not select a configuration merely
because it stays more upright.

The driven bodies finish with upright-axis projections 0.222–0.372, worse than
the two nearly upright residents in the prior no-coxa run. Late horizontal path
is 0.002–0.355 mm and is not evidence of walking while tipped. Maximum named
antagonist rate separation is 0.00105. The driven/frozen root difference reaches
0.186 mm, without establishing beneficial feedback.

Verification passed: all recorded sensory offsets and muscle coefficients
reconstruct exactly, unsupported channels exert zero force, and zero-drive qpos
matches the previous zero-drive trace exactly. The receipt is
`evidence/coxa-verification-001.json`. All 18 Python unit tests passed. No
simulation remains running after these finite trials.

Use `verify_coxa_muscles.py --source-scene ORIGINAL_WORLD_JSON --output FRESH_JSON`
to authenticate receipts, reconstruct sensory offsets and muscle coefficients,
check unpowered actuator boundaries and compare zero-drive mechanics with the
earlier zero-drive trace. It does not prove biological fidelity.

## Reproduction

Run the mapper against the pinned original scene with MuJoCo 3.12.0 and retain
its JSON. Build the existing physics-step host as described in PHYSICS_MUSCLES.md.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_coxa_muscles.py \
  --source ../chreatures-muscle-step \
  --model ../releases/chreatures-v5-garden/model \
  --scene ../scenes/no-joint-passives/world.json \
  --motor motor-annotations.json --sensory proprioceptor-annotations.json \
  --sensory-polarity 1 --coxa-map evidence/coxa-map-001.json \
  --output runs/coxa-driven --steps 200
```

Use fresh output directories for `--condition frozen-sensory` and
`--condition zero-motor`. The goal remains useful, biologically grounded whole-fly
behavior; this incomplete actuation model does not satisfy it.
