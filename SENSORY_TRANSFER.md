# Where the small motor output originates

The first post-reassessment diagnostic replays one resident's four seconds of
physical senses from `physics-muscle-none-driven-001` through the unchanged CNS.
It compares body-afferent current gains 0, 1, 10 and 100 around neutral current.
Vision is identical across lanes and external context is zero. Currents are
bounded to [0,1]. These gains are diagnostic perturbations, not physiological
calibration or settings installed in the embodied system.

The ordinary-gain lane reproduces recorded motor rates to a maximum error of
1.49e-8. This supports the replay comparison. It does not validate the biological
model. All state fields were finite. Four simulated seconds took 8.08 wall
seconds on one GPU, excluding model setup.

| Body-current gain | Maximum individual motor change versus neutral body input | Maximum named antagonist separation | Maximum fraction of body currents clipped |
| --- | --- | --- | --- |
| 0 | 0 | 0.00000240 | 0% |
| 1 | 0.00416 | 0.000521 | 0% |
| 10 | 0.0204 | 0.00264 | 25.6% |
| 100 | 0.0301 | 0.00531 | 56.7% |

Antagonist separation is the absolute difference between the mean flexor and
extensor motor rates for the twelve currently actuated hinges, before recruitment
filtering. Rates are normalized model quantities, not spikes/s. The maxima can
occur at different times and in different neurons/joints. They are not gains of
a fitted linear transfer function.

## Interpretation and next action

The body sensory path is not dead. It accounts for substantially more motor
variation than vision alone on this recorded trajectory. Increasing its current
does increase differentiated output, but a 100-fold increase gives only about
ten times the maximum named antagonist separation and heavily clips afferents.
This is not a justification for installing a global gain increase.

Equal-mean antagonist recruitment leaves most muscle activation as shared tonic
drive near the model's 0.2 baseline. With the present contractile coefficient
0.03, even the largest 100x separation contributes only about 0.000159 normalized
force before filtering, excluding stiffness and damping. That is an interface
scale observation, not a claim that a particular force would produce walking.

The next implementation should replace the broad, uncalibrated sensory projection
and examine anatomically constrained recruitment for the supported leg pathways,
using the existing named proprioceptor assay as a constraint. Do not invent a
gait or subtract baseline merely to manufacture motion. Continue toward coherent
six-leg control, not an ever-expanding gain sweep. The current CNS is responsive,
but biological correctness and useful full-body coordination remain unproven.

## Reproduce and inspect

`probe_sensory_transfer.py` requires `--source`, `--model`, `--trial` and a fresh
`--output` directory. Use the pinned source/release and the source trial above.
It changes no model arrays or physical host configuration. The replay is open
loop: amplified lanes receive recorded senses, not consequences of their own
motor outputs. It cannot establish beneficial feedback or autonomous behavior.

Completed output is `evidence/trials/sensory-transfer-002`. The first invocation's
trace and report failure are retained in `sensory-transfer-001`. A source-trace
axis-selection bug affected its final comparison, not neural integration.

`analyze_sensory_transfer.py` independently authenticates source and runner hashes,
checks numeric finiteness, recomputes replay error and extracts named antagonist
separations from the existing muscle cohorts. Its arguments are `--trial`,
`--source-trial`, and a fresh `--output`; the committed receipt is
`evidence/sensory-transfer-analysis-001.json`. NPZ files use numeric arrays only.
