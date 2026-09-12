# Passive-mechanics ablation and force-update failure

`make_passive_variant.py` creates fresh, authenticated scene variants without
changing the original source. The first sets the 504 explicit joint stiffnesses
to zero and retains damping 0.5. The second also sets damping to zero. These
are diagnostic removals, not proposed biological parameter values. Mesh hashes,
geometry, initial pose, muscle definitions and sensory metadata remain fixed.
The variant receipts preserve each changed joint and both source/derived hashes.

Four-second trials on Vast used the unchanged full-CNS and aggregate-muscle
runner, with enabled versus zero motor input for each mechanical condition:

| Mechanics | Driven | Zero motor |
| --- | --- | --- |
| Original stiffness/damping | Completed | Completed |
| No joint stiffness | Completed | Completed |
| No joint stiffness or damping | Failed at about 0.03 s | Completed |

The original driven bodies ended with thorax heights 0.73–0.80 mm; removing
springs lowered them to 0.53–0.61 mm. Last-second horizontal paths increased
from roughly 0.006–0.010 mm to 0.028–0.119 mm. This is changed settling/drift,
not demonstrated walking. Zero motor input also changes posture and motion.
With both stiffness and damping removed, the zero-motor bodies tilt strongly;
their final thorax-up components are 0.22–0.56. The driven run instead triggered
MuJoCo's huge-acceleration warning at 0.0296 s and the native clock/finite guard.
Its incomplete trace and failure receipt are preserved, not counted as behavior.

The original driven repeat agrees with the previous two-second trial over its
shared prefix: maximum thorax-position difference 9.90e-11 mm, motor-rate
difference 1.49e-08, and command difference 4.66e-10.

## Isolated timing experiment

`probe_muscle_timestep.py` replays the failed run's activation prefix in a static
copy of its physical scene, holding the last available activation afterward.
Holding pose/velocity-derived torque for 10 ms fails at 0.032 s. Re-evaluating
that force every 0.1-ms physics step completes the 0.2-s test, as does zero
force. This supports force-update timing as a numerical failure mechanism.
It does not establish long-run stability or numerical convergence. The replay
omits native aerodynamics/ecology and uses unscaled original force capacity;
it is not a whole-CNS repair or biological-behavior demonstration.

The direct implementation consequence is to evaluate muscle mechanics at the
physics timestep while updating neural drive at the neural timestep. That
repair has NOT yet been integrated into the native whole-CNS runner.

## Limits and a relevant biological reference

`audit_body_contact.py` reconstructs thorax/abdomen terrain contact pairs from
recorded poses. It reports no non-positive-distance core contacts in the two
ablated zero-motor runs. This is not a native force measurement: positive
contact margins, other body parts and online geometry changes are not included.
It cannot by itself establish standing or prove that the body is unsupported.

Wang et al.'s [reviewed preprint](https://elifesciences.org/reviewed-preprints/107390v1)
provides passive-joint measurements and motor-silencing experiments in female
flies. It reports that passive forces alone cannot support the animal and
discusses an approximately 100-ms active-force decay. This offers an external
validation target, not direct permission to copy values into this model: joint
coordinate mappings, units, specimen differences and activation/deactivation
interpretation still need checking. No parameters from that paper have been
installed here.

All trials, including the failure, are under `evidence/trials/passive-*`;
the timing probe is `evidence/trials/muscle-timestep-001`. These experiments
diagnose the existing prototype, not achieve the rich whole-fly objective.
