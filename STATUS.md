# MaleCNS embodiment experiment

Latest integration: a separate free-body six-leg muscle prototype runs full-CNS
proprioception with 90 muscle units, 78 mapped to named motor pools. The corrected
single-floor trial completes two seconds; physical checkpoint replay is exact
on 30 sampled intervals. Thorax/abdomen contacts carry ground forces, so it is
resting on its body, not standing. Initial duplicate-floor construction is retained
as a flawed trial. See [SIX_LEG_BODY.md](SIX_LEG_BODY.md). Rich behavior is unachieved.

Latest muscle integration preparation: a shared named-pool approximation connects
all 15 published foreleg muscle units to 39 motor neurons. Static directional
coverage increases from rank five to seven, including shared-pool constraints.
Full-CNS foreleg feedback still settles. Source/target joint topology differs,
so direct index copying into the whole-body rig is invalid. See
[POOLED_MUSCLES.md](POOLED_MUSCLES.md). This does not resolve whole-body support yet.

Latest diagnosis: static gravity/contact balance at eight recorded upright poses
is infeasible with the current 18-coordinate actuation under original limits;
six cases remain infeasible with unlimited torque (two numerically inconclusive).
All-42-coordinate solutions exist under the diagnostic's ideal-lock assumptions.
See [STATIC_SUPPORT.md](STATIC_SUPPORT.md). Complete missing mechanical support
before interpreting further collapse as CNS failure. Runtime settings unchanged.

Latest actuation update: 34 named coxa rotator neurons now drive fixed,
geometry-checked protraction/retraction coordinates across six legs. Coverage
is 18 of 42 leg channels. Driven/frozen/zero two-second trials completed;
driven bodies still tip, with worse final posture than some previous trials.
This is not behavioral improvement. See [COXA_MUSCLES.md](COXA_MUSCLES.md).

Latest implementation: named leg-position transduction replaces the random
projection onto 91 claw afferents spanning six legs, with one missing subtype
explicitly retained as a gap. Both polarity hypotheses and frozen/zero controls
completed two seconds. Numeric reconstruction and 16 unit tests passed; no
useful behavior demonstrated. See [NAMED_LEG_SENSES.md](NAMED_LEG_SENSES.md).

Subsequent diagnostic: four-second physical sensory replay reproduces motor
rates within 1.49e-8 and separates body-driven from vision-only output. Stronger
body-afferent currents increase motor responses but saturate inputs while named
antagonist separation remains small. See [SENSORY_TRANSFER.md](SENSORY_TRANSFER.md).
No neural parameter or embodied controller setting was changed by this assay.

Latest update: the physics-step muscle repair passed bounded full-CNS trials
(four seconds driven, two seconds each zero-motor and frozen-sensory). No useful
behavior was demonstrated. See [PHYSICS_MUSCLES.md](PHYSICS_MUSCLES.md) for evidence
and the actual artifact's heterogeneous CNS parameter audit, and
[REASSESSMENT.md](REASSESSMENT.md) for revised priorities. Entries below preserve
the chronological investigation; no simulator is currently left running.

Objective: physical environment → sensory organs → MaleCNS dynamics → motor
neurons → muscles/body → changed sensory input, with rich emergent fly behavior.
No scripted behavior selector or separate action policy qualifies as success.

## Source audit (2026-09-12)

- Chreatures c30fbccb541ae5bc2cbdb88f7ef38848c611a8e6: full curated
  165,122-neuron graph; Torch CNS V5 reference provides a CUDA port candidate.
  Its 92 outputs depend on 815 motor neurons. However, the decoder is a learned
  masked joint-servo interface, NOT a measured muscle recruitment map.
  Its separate private resident supplies descending-neuron context; disable this
  by supplying zero context in the strict experiment. Initialized published
  weights do not establish useful locomotion. Even trained upstream assessments
  only show short upright survival, not rich behavior.
- FlyBrain cdd3a127766ec184e19c4988fe12b6fd2cbc64fd: stronger demonstrations,
  but direct odor guidance and engineered gait/flight stabilization are not
  evidence of behavior emerging solely through MaleCNS motor neurons.

## Machine

Existing 4×3090 Vast host is reachable; old experiment is removed. Official
MaleCNS annotation, transmitter and edge files exist under /opt/malecns/data/raw.
New isolated Python 3.12 environment has NumPy, SciPy, Arrow and MuJoCo.
CUDA Torch 2.8.0+cu128 is installed; pinned source and release are transferred.
No running embodied fly and no demonstrated behavior yet.

## Verified neural execution

`probe_cns.py` authenticates every released tensor against its transport and raw
hash, validates the upstream array contract, and runs the upstream Torch V5
recurrence on GPU0. Published archive SHA256:
505456e9b214bb23865c8193c5af441d0409a36ca3634d0b5c0155ac958977e4.

Remote receipt: /opt/malecns/neural-probe-001.json. Two neural states (dark/light),
zero external context, 50 steps = 0.5 simulated seconds each, 1.687 wall seconds,
1,666,230,784 peak allocated CUDA bytes, all fields finite. Maximum light/dark
rate difference 0.1860; motor difference only 2.876e-6. This is a synthetic neural
diagnostic, NOT a physical closed loop or behavioral competence demonstration.
The initialized motor readout is nearly insensitive in this test.

Next integration location identified:
`research/fly_learning/native_host.py` (NativeActualFlyWorld, TorchFullCNS),
`native/fly-world` (Rust native MuJoCo host), and
`research/fly_learning/native_assessment.py` (closed-loop assessment).
Audit these and use their physical bridge with no private resident controller.
Native host declares MuJoCo 3.12.0; environment currently has 3.13.0, so check
and pin exact build compatibility before compiling rather than changing receipts.

## Next evidence required

1. Authenticate published CNS tensors and reproduce full-graph CUDA recurrence.
2. Join actual body sensors to afferents and motor activity to physical actuation,
   with zero external context and no behavior-controller bypasses.
3. Record physical trajectories, senses and neural/motor state; compare intact,
   sensory-disconnected and motor-disconnected runs.
4. Establish multiple meaningful behaviors and transitions across varied starts.
5. Audit and improve muscle/sensory/dynamics assumptions against biological
   evidence. A learned servo adapter is an explicit approximation, not ab initio
   muscle physiology. Keep failed trials and do not relabel upright-only success.

Raw wiring does not supply all physiological parameters or a measured complete
neuromuscular interface. This remains an integration/research task, not a claim
that installing an existing demo fulfills the objective.

## Physical loop and ablations (2026-09-12)

Native Rust host compiled with locked Cargo dependencies and MuJoCo 3.12.0.
`run_embodied.py` joins authenticated V5 tensors to protocol v2 world samples,
checking sensory, actuator and body-schema identities. Only optic/BODY807 enter
the CNS; external context stays zero, and only motor-neuron-derived MOTOR92 is
sent to the physics host. Records retain senses, motor neuron rates, requested
and delivered motor commands, joint positions, root positions and uprightness.
No teacher/private resident controller is instantiated.

Completed 200-step (2-second), four-resident trials:
- /opt/malecns/trials/intact-002 (GPU0, ~50.7 wall seconds)
- /opt/malecns/trials/sensory-disconnected-001 (GPU1)
- /opt/malecns/trials/motor-disconnected-001 (GPU2)
All are terminal, no simulation is currently left running. The failed
intact-001 remains: a recording index dtype error stopped it before any advance;
the corrected runner casts motor row indices to long.

`compare_trials.py` produced /opt/malecns/trials/comparison-001.json. Bodies
remained upright in ALL conditions. Initial ~1 mm displacement was predominantly
vertical settling, not walking. After the first second, intact horizontal path
was only ~0.0024–0.0025 mm per resident. Sensory disconnection changed root
trajectories by at most 0.00379 mm; motor command clamping by 0.00188 mm.
Maximum intact joint command magnitude was only 6.97e-5. These are evidence
AGAINST claiming useful behavior from this initialized model.

Important: `motor-disconnected` means neural commands clamped to zero, NOT
muscle paralysis. Upstream position servos still hold neutral joint targets.
This intrinsic servo feedback is a substantial nonbiological shortcut, even
without an explicit high-level controller. Do not claim uprightness is neural.
Four residents in one shared world are not four independent trials.

## Next mechanistic component

FlyMimic source audited at /home/ubuntu/malecns-flymimic-audit,
commit 9ea1131626cd76f7203b74076ef8f0e9cab30bef (no AGENTS.md).
https://github.com/gizemozd/FlyMimic
https://arxiv.org/abs/2509.06426
Contains anatomy-derived Hill-type leg muscle models converted to MuJoCo.
Inspect/reuse physical assets, NOT its PPO imitation policy. The paper's model
omits some muscles and its demonstrations are not free-moving whole-fly CNS
behavior. Original parameter work: https://github.com/gizemozd/neuromechfly-muscles
Next: inspect muscle assets and map measured MaleCNS/MANC motor identities to
muscle targets, preserving unresolved mappings rather than inventing them.

Relevant caution: https://faculty.washington.edu/tuthill/docs/TheSphinx_2026.pdf
shows realistic fly walking can be learned through an implausible worm-to-fly
decoder. Behavioral appearance alone cannot validate our neural/body interface.

## Muscle component execution and mapping audit

`audit_motor_annotations.py` authenticated the raw annotations and exported
all 815 motor neurons with exact body IDs, descriptive labels and full-CNS row
indices. Output: /opt/malecns/motor-annotations.json (also copied locally).
The row convention is all 165122 Traced neurons sorted by bodyId.

FlyMimic physical assets + license deployed at /opt/malecns/flymimic.
`probe_muscles.py` ran all 15 left-foreleg Hill-type actuators, separately,
comparing 0.05 activation against zero command from the identical keyframe
for 100 physical substeps (0.01 seconds). All 15 yielded finite state and
nonzero joint responses. Six anatomical-name candidate correspondences are
listed in muscle-probe-002.json; remaining nine are unresolved rather than
arbitrarily distributing neuron activity across muscle subdivisions.
Candidate correspondences remain cross-specimen inferences, not measured
neuromuscular recruitment weights. Do not claim these are validated junctions.

The model has NO free body joint, only left-foreleg muscle actuation, and locks
the opposite foreleg. Current FlyGym's muscle wrapper explicitly has the same
limitation: https://neuromechfly.org/api_reference/flygym/compose/fly/musculoskeletal/
Therefore it cannot replace the whole-body model without further physical
integration. This is a component test, not a smaller substitute goal.

First muscle-probe-001 computed/wrote its result but its console summary failed
on numpy-int JSON serialization. Fixed casting; muscle-probe-002 exited cleanly.
Original muscle-development repository linked from FlyMimic:
https://github.com/gizemozd/neuromechfly-muscles returned Repository not found
on unauthenticated clone. No credentials requested or transferred.

The MANC primary paper supports motor target assignments through inter-specimen
matching and states important exceptions to serial leg homology:
https://elifesciences.org/articles/96084 (Leg MNs, Figure 7).
Next work must resolve whole-body muscle geometry/actuation and remaining
sensory/motor physiology, rather than training a behavior decoder. Existing
whole-body baseline and all failed/negative trials remain intact.

## Named efferent interface implemented

`neuromuscular.py` implements NamedMuscleDrive, a fixed anatomical support
matrix from the six name-matched motor-neuron cohorts to their muscle targets.
It receives ONLY normalized motor rates; no target joint angles, sensory/world
state, time, reward, or behavior labels. Equal averaging within a muscle cohort
is an explicit UNCALIBRATED recruitment hypothesis. Unresolved muscles have zero
neural drive. Four unit tests passed on Vast (isolation, zero input, nonmotor
rejection, invalid-rate rejection).

`replay_motor_to_muscle.py` authenticated the CNS model and the earlier intact
trace, then drove those muscles using recorded motor rates for one resident.
Paired zero-neural-drive replay started from the same physical keyframe.
/opt/malecns/trials/muscle-replay-001/result.json: completed 200 steps/2 seconds,
finite state, maximum joint difference 0.5306746338 radians. This establishes a
candidate efferent component ONLY. It is open-loop replay: the muscle movement
does not feed the source neural state. No behavioral competence is demonstrated.

Checked original OpenSim `best_combined_full.osim`: its 15 muscles are likewise
left-foreleg only. The authors' project site's ground-walking example explicitly
uses direct torque control for middle/hind legs:
https://gizemozd.github.io/fly_mimic/ (Ground Locomotion).
No complete six-leg muscle asset was found in this release.

Also inspected original NeuroMechFly spring-damper muscle implementation:
https://github.com/NeLy-EPFL/NeuroMechFly/blob/main/NeuroMechFly/control/spring_damper_muscles.py
Its active inputs use sin(oscillator phase); using that controller unchanged
would prescribe locomotor rhythm outside MaleCNS. The passive/contractile law
could be reused as an explicitly approximate actuator, but not its oscillator
policy. Do not equate generic joint antagonist pairs with identified fly muscles.

## Sensory audit and public repository

Public source/evidence repository requested by user:
https://github.com/deepai-org/malecns-embodied . Maintain subsequent fly work
there, including negative results; never publish credentials or connection keys.

`audit_proprioceptors.py` exported 425 annotated chordotonal neurons with exact
CNS rows. Left foreleg ProLN includes one SNpp50 and three SNpp51 neurons.
The MANC annotation paper identifies these as claw subtypes with different
predicted effects on tibia flexor/extensor circuitry:
https://elifesciences.org/reviewed-preprints/97766v1 (Figure 59 discussion).
This does NOT yet identify their exact physical tuning curves or justify
assigning sensory gain/polarity based on desired behavior. No sensory adapter
has been changed yet. Existing joint-to-sensory adapter uses broad leg cohorts.
Next useful assay: test separately stimulating these named sensory subtypes
against the predicted motor-neuron effects in the full graph before inventing
a position-to-current mapping.

## Full-graph proprioceptor assay

Implemented `probe_proprioceptive_circuit.py` and executed amplitudes 0.05, 0.2,
0.5 in five independent states per run (control plus four subtype perturbations).
All four subtypes produced expected relative flexor/extensor preferences at all
three amplitudes. SNpp51 nevertheless excited, rather than inhibited, the
extensor motor group. This is partial anatomical-circuit agreement, not reflex
or behavior validation. Inputs are arbitrary model currents, not calibrated
physical joint stimuli. Detailed evidence, limitations and reporting corrections
are in PROPRIOCEPTION.md; original artifacts are retained.

## First physical muscle feedback loop

Implemented `run_muscle_feedback.py`: fixed-base foreleg angle feeds candidate
SNpp50/51 claw-population currents, through the full CNS, back into the existing
six named muscle cohorts. Two opposite position-tuning hypotheses are retained,
each with frozen-sensory and zero-motor controls. No behavioral controller or
learned motor decoder participates. Sensory transduction remains uncalibrated
and bypasses organ strain mechanics; nine muscles remain unresolved.

Executed six-lane 2-second and 10-second trials on Vast. Sensory feedback affects
both neural rates and subsequent physical motion, but the maximum joint effect
relative to frozen sensory input is under 0.000867 radians. The driven foreleg
largely settles rather than generating a sustained motor pattern. All trace
values remained finite; recorded sensory transforms and motor-to-muscle commands
were verified directly. Six unit tests pass locally and remotely. See
MUSCLE_FEEDBACK.md and evidence/trials/muscle-feedback-{001,002} for assumptions,
controls, runner provenance and raw numeric traces.

This closes a previously absent component feedback path, not the requested
whole-fly loop. Priorities remain a defensible whole-body muscle interface,
explicit sensory-organ transduction and neural dynamics that reproduce more
than qualitative circuit preferences. Keep these partial assays subordinate
to the whole-body objective; settling in a fixed foreleg is not success.

## Whole-body actuator boundary and aggregate muscle trial

Built a separate patched Chreatures host with experimental `advance_torque`.
Original source, baseline binary and default position-control operation remain
unchanged. The patch is public, hashed in sources.json, and adds measured
actuator forces to samples. Zero torque gives exactly zero actuator force;
an isolated pulse powers only its requested joint. The old zero-position
command still produces up to 3.0078 model force units in the matched host test.

Added 24 approximate antagonist muscle units spanning twelve hinges across
all six legs, driven only by named Tr/Ti motor-neuron cohorts. Ran intact,
zero-motor, frozen-sensory and reverse-sign trials for two seconds on Vast.
All completed, but mostly settled: intact final-second horizontal paths are
only 0.0064–0.0117 mm. Max thorax differences from intact are 0.00613 mm for
zero-motor and 0.00344 mm for frozen sensory input. No rich behavior demonstrated.

Verified all recorded muscle commands exactly, finite traces and exact zero
force on unsupported actuators. The zero-motor body nevertheless remains
upright. A physical XML audit identifies 504 non-free joints with stiffness 10
and damping 0.5, including fixed neutral leg spring references. These built-in
passive mechanics remain even after servo removal and must be tested explicitly
before attributing posture or weak responsiveness to neural dynamics alone.
See AGGREGATE_BODY.md, the public patch and evidence/aggregate-body-comparison-001.json.

## Passive-mechanics diagnostics and route reassessment

Executed four-second original versus spring-free trials with driven and zero
motor input. All four completed; removing springs changes settling but produces
no demonstrated walking. Removing damping as well leaves the zero-motor run
finite but makes the driven run fail at about 30 ms. A separate physical replay
reproduces failure with 10-ms held torque and completes 0.2 s when muscle force
is recomputed every 0.1-ms physics step. This identifies a concrete numerical
repair to implement, not a recovered biological behavior.

See PASSIVE_MECHANICS.md for controls and limitations, including the failed run.
The user requested a reassessment of the overall route. REASSESSMENT.md records
why further component sweeps alone are too indirect: most motor output remains
tonic, leg actuation is incomplete, and CNS behavioral feasibility is unproven.
The full objective remains unchanged. No new unattended experiment is running.
