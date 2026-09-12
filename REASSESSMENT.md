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
this initialized graph longer. This reassessment is a recommendation and record
of the evidence; it does not claim the numerical repair or interface completion
has already been implemented.
