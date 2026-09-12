# Recovering the published muscle model's directional coverage

The previous exact-name interface connected six of fifteen muscle-tendon units
in the FlyMimic foreleg asset. A more permissive, explicitly approximate named-pool
mapping now connects all fifteen units to 39 MaleCNS motor neurons. It reuses
the published geometry and muscle parameters; no gait policy is imported.

This addresses a real limitation: at the source keyframe, the six-muscle subset
has torque-matrix rank five across seven foreleg coordinates. The fifteen-unit
model has rank seven. A nonnegative torque-direction test reaches both signs of
all seven coordinate axes, including when the shared motor-pool weights are
applied. The six-unit subset reaches only one of those fourteen pure directions.

These are **directional coverage** results, not sufficient-force or behavior
results. The test permits unbounded nonnegative activation/rates and tests one
fixed pose. The runtime still bounds normalized rates to [0,1], and the CNS may
not generate the independently chosen motor-rate patterns used in this audit.

## Recruitment assumptions

`pooled_muscle_mapping.py` preserves the six earlier name correspondences and
adds coarse assignments for the remaining units:

- Promotor subdivisions share the combined tergopleural/pleural promotor pool.
- Trochanter-flexor subdivisions share the trochanter-flexor pool.
- Composite sterno/tergo-trochanter extensors share the union of those named pools.
- The modeled tibia flexor/extensor receive their corresponding named pools.

Every selected neuron is restricted to the left foreleg. Subdivisions receive
the same mean rate where their pool is shared. This is not a measured assignment
of individual motor units to fibers. Muscle capacities are unchanged; duplication
of a pool does not imply duplicate anatomical innervation is established.

The [authors' model description](https://gizemozd.github.io/fly_mimic/) identifies
15 units covering 12 of 19 anatomical groups and explains that unknown muscle
parameters were optimized against recorded limb kinematics. We use those
mechanical parameters, not their PPO controller. The source is foreleg-only;
its ground-walking example uses direct torque control for the other legs.
Thus this is neither complete fly musculature nor an ab initio behavior demo.

## Physical feedback result

`run_pooled_feedback.py` runs the full CNS in the existing fixed-base foreleg
loop. Both sensory polarities have feedback, frozen-sensory and zero-motor lanes.
All six lanes completed two seconds. Maximum feedback/frozen angle differences
are 0.000730 and 0.000870 rad; feedback/zero differences are about 1.135 rad.
The maximum joint range over the final second of feedback is about 0.000114 rad.
This is chiefly a tonic muscle response and settling, not useful regulation.

`verify_pooled_feedback.py` checks model, mapping and source hashes; reconstructs
every requested control within 2.99e-8; checks zero commands exactly; and
reconstructs the sensory position transform. It also recomputes the rank-seven
raw torque matrix. It does not independently rerun all direction optimizations.

Trial `pooled-feedback-001` used the old runner successfully, but its hardcoded
assumption text incorrectly continued to say six muscles/nine unresolved. That
receipt is retained with this correction. Trial `pooled-feedback-002` reruns the
same calculation with mapping-independent explanatory text and a separately
hashed runner. Neither source nor earlier evidence was overwritten.

## Whole-body integration boundary

| Joint group | Source foreleg asset | Current whole-body rig |
| --- | --- | --- |
| Thorax–coxa | Three coordinates | Three coordinates |
| Coxa–trochanter/femur | Three coordinates | Two coordinates |
| Femur–tibia | One coordinate | One coordinate |
| Tibia–tarsus | Absent | One coordinate |

Equal total dimension does not make these joint spaces interchangeable. The
source's extra trochanter coordinate cannot be relabeled as tarsus motion.
Integration must preserve actual attachment geometry and handle distal passive
or tendon mechanics explicitly. Simply replicating foreleg geometry onto middle
and hind legs would also be an unvalidated serial-homology approximation.

The next implementation target is to carry multi-axis muscle geometry into a
coherent whole-body mechanical model, rather than add more independent torque
hinges or install the static-support optimizer as a controller. This fixed-base
run is preparation for that integration, not a substitute completion goal.

## Reproduce

Generate a fresh mapping with `pooled_muscle_mapping.py --inventory
muscle-probe-002.json --annotation motor-annotations.json --output FRESH_JSON`.
Use `run_pooled_feedback.py` with the same arguments as `run_muscle_feedback.py`,
but pass the new mapping. `audit_muscle_directions.py --help` exposes the static
audit. The pinned source model must use MuJoCo 3.12.0.

Evidence: `evidence/pooled-muscle-mapping-001.json`,
`evidence/muscle-directions-001.json`, `evidence/trials/pooled-feedback-002`, and
`evidence/pooled-feedback-verification-001.json`. No simulation is left running.
