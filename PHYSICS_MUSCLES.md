# Physics-step muscle forces

The previous spring-and-damper-free driven run failed at about 0.03 seconds.
Its velocity-dependent muscle torque was evaluated once per 10 ms neural tick
and then held across 100 physics steps. The new host reevaluates that same
force law from physical joint position and velocity every 0.1 ms. Recruitment
still updates every 10 ms. No muscle gain, neural parameter or behavior policy
was changed.

## Evidence

All runs use the authenticated no-joint-passives scene and four residents.
These are shared-world residents, not independent replicates.

| Trial | Duration | Outcome |
| --- | --- | --- |
| physics-muscle-none-driven-001 | 4 s | Completed, finite; no useful behavior demonstrated |
| physics-muscle-none-zero-001 | 2 s | Completed; trajectory exactly matches previous zero-force prefix |
| physics-muscle-none-frozen-001 | 2 s | Completed with sensory inputs held at initial values |

The driven run took 153 wall seconds. Bodies still tip or settle; initial
displacement is not locomotor competence. Driven versus frozen sensory root
positions differ by up to 0.121 mm over two seconds. This establishes sensitivity,
not beneficial feedback: unstable contact and tipping can amplify tiny changes.

The host contract probe checks zero force, rejection of negative stiffness and
NaN without time advancement, and exact agreement with the old torque mode for
one constant-force tick. The trace verifier reconstructs every recorded muscle
coefficient from motor activity with zero error and checks that unsupported
actuators exert zero force. Fourteen Python unit tests passed, and the locked
Rust release build succeeded with MuJoCo 3.12.0.

Receipts and numeric traces are under `evidence/trials/physics-muscle-*`;
`evidence/physics-muscle-verification-001.json` records the independent checks.
This is a bounded numerical repair, not a timestep-convergence proof, calibrated
muscle model or demonstration of walking.

## Interface and reproduction

Apply `patches/experimental-muscle-step.patch` to a fresh checkout of the pinned
original Chreatures revision. It includes the older torque extension: **do not
apply both patches**. Build as described in README.md. Original source, torque
host and their trial records remain unchanged. Hashes are in `sources.json`.

The new `advance_muscle` RPC takes `muscle252_base64`: 84 triples per resident,
little-endian float32 `[constant, stiffness, damping]`. Each physics step uses
`clamp(constant - stiffness*q - damping*qvel, -1, 1)` through the force-clamped
actuators. This cancels the original affine position-servo forces. Coefficients
are finite and bounded; negative stiffness/damping is rejected. The separate
`advance` and `advance_torque` interfaces remain available.

Generate the zero-stiffness/zero-damping scene with `make_passive_variant.py` as
in PASSIVE_MECHANICS.md, then run:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_physics_muscles.py \
  --source ../chreatures-muscle-step \
  --model ../releases/chreatures-v5-garden/model \
  --scene ../scenes/no-joint-passives/world.json \
  --motor motor-annotations.json --output runs/physics-driven --steps 400
```

Use fresh output directories and `--condition zero-motor` or
`--condition frozen-sensory` for controls. The verifier uses committed trial
names and requires `--source-scene` pointing at the pinned original scene and
a fresh `--output` path. `probe_physics_muscle_host.py --help` exposes the bounded
native contract test. No background process is installed.

Remaining assumptions include only 12 of 42 leg channels actuated, aggregate
antagonist geometry, uncalibrated motor recruitment and broad sensory mapping.
Native aerodynamics and contact mechanics remain. Ecological bookkeeping receives
zero high-level commands and was designed for servos: no validated muscle energy
budget is claimed.

## CNS operating-point clarification

`audit_cns_operating_point.py` authenticates the actual release arrays, without
tuning them. `evidence/cns-operating-point-001.json` shows baseline rate 0.2,
recurrent gains 0.697–1.549 and neuronal time constants 32–158 ms. The latter two
are heterogeneous, unlike the uniform initializer defaults. The fast recurrent
gain-weighted row-sum bound is 1.549, not below one; it cannot establish global
contraction, and does not cover the full adaptive/modulated embodied system.

At exactly neutral input, initialized state is a deterministic fixed point.
That does not prove incapacity under sensory drive. In the pinned upstream code,
`research/anatomical_cns/export.py` copies earlier dynamic parameters into the
new artifact while initializing new interfaces; `upgrade_v4_to_v5.py` preserves
them. The `initialized-untrained` metadata is not evidence that every parameter
is fresh or uniform. Exact training-run lineage remains unresolved. CNS
feasibility tests should distinguish weak sensory transduction, weak neural
responses and weak motor-to-muscle scaling before changing the neural model.
