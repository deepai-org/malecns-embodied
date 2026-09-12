# Route reassessment

The goal remains a rich, biologically grounded whole fly—not an isolated reflex,
a scripted gait, or a connectome decorating a learned behavioral policy.

The present sequence has become too focused on component diagnostics. Those
tests exposed real flaws, but more of the same is not the shortest credible
route to the goal. The current full graph is a wiring-constrained, initialized
rate model, not a validated operational nervous system. In the four-second
original-mechanics driven trial, all recorded motor rates span only about
0.19920–0.20083. Late opposing muscle activations differ by at most 0.000113.
Most current output is tonic co-contraction, not differentiated motor activity.

Actuation is also incomplete: only twelve of 42 leg-actuator channels have
named aggregate muscles. Passive support elsewhere cannot substitute for the
missing muscles. Changing neural parameters before resolving this interface
would confound a neural failure with an inability of the body to execute it.

Recommended priorities:

1. Fix the identified force-update timing problem once, verify it, and freeze
   the basic simulation infrastructure. Avoid open-ended passive-parameter sweeps.
2. Make CNS feasibility the main gate: require stimulus-dependent, structured
   motor output against biological perturbation evidence before more environment
   work. Examine or replace the initialized dynamics if they cannot produce it;
   preserving this particular simulator is not part of the user's objective.
   Test this before investing in further custom anatomical detail.
3. Complete a coherent six-leg motor/muscle and sensory interface using existing
   anatomical assets and documented approximations. Do not require every missing
   parameter to be measured, but distinguish assumed values from evidence.
4. Use the full embodied fly in one environment with a small, fixed battery
   testing movement initiation, sustained coordination, orientation and recovery.
   These are milestones toward behavioral breadth, not a reduced completion
   criterion. Keep the same measurements across candidate implementations.

Parameter fitting to biological data and mechanistic approximation are allowed;
rewarding a trajectory or adding a behavior selector is not equivalent evidence.
No current result justifies expecting rich behavior to emerge merely from running
this graph longer.

Update: priority 1's force-update repair is implemented and checked over four
seconds, with matched two-second controls; see [PHYSICS_MUSCLES.md](PHYSICS_MUSCLES.md).
That is not a convergence proof or behavioral improvement. Interface completion
and the CNS feasibility gate remain outstanding.

The operating-point audit also sharpens the diagnosis: the actual artifact has
heterogeneous recurrent gains and time constants, unlike the uniform initializer
defaults. Its exporter inherits earlier parameters while initializing new
interfaces. The release's `initialized-untrained` label must not be interpreted
as every parameter being freshly uniform. Exact training lineage is unresolved;
neither that label nor near-tonic embodied output proves the CNS cannot generate
richer dynamics. Test sensory input strength, CNS responses and neuromuscular
scaling separately before replacing or increasing neural gains.

Subsequent whole-body integration added named claw transduction and six coxa
rotator coordinates, without useful behavior. The [static-support test](STATIC_SUPPORT.md)
now gives a concrete reason to prioritize coherent mechanics: existing actuation
cannot balance sampled upright poses even under favorable assumptions, whereas
all leg coordinates can. Avoid more neural-gain or partial-actuation behavioral
sweeps until missing joint reactions have a defensible physical implementation.
