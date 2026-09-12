# Named proprioceptive circuit assay

Question: does the full initialized MaleCNS recurrence preserve the qualitative
motor preferences expected from identified sensory subtypes?

The hypothesis comes from the MANC anatomical circuit analysis in
[Marin et al., Figure 59](https://elifesciences.org/reviewed-preprints/97766v1).
It is a cross-specimen circuit prediction, **not measured MaleCNS physiology**.

`probe_proprioceptive_circuit.py` runs five independent neural states: an
unstimulated control and one for each left-foreleg ProLN subtype SNpp50, SNpp51,
SNpp39 and SNpp41. Positive current is injected only into their annotated
afferent rows, from 0.3 to 1.0 model seconds. No body, behavioral policy, learned
motor decoder or external context participates. Raw motor rates are compared
with the matched control lane at every timepoint.

At model-current amplitudes 0.05, 0.2 and 0.5, all four subtypes have the expected
**relative** flexor/extensor preference. This is not equivalent to reproducing
both excitation and inhibition. At amplitude 0.2, late mean rate changes are:

| Input | Tibia flexor | Tibia extensor | Interpretation |
| --- | ---: | ---: | --- |
| SNpp50 | −0.000115 | +0.000118 | Extensor preference; delayed effects untested |
| SNpp51 | +0.001557 | +0.000105 | Flexor preference, but extensor inhibition fails |
| SNpp39 | +0.000163 | −0.000087 | Expected opposing signs |
| SNpp41 | −0.000002 | +0.000002 | Expected opposing signs, very small response |

These are normalized rate-model values, not Hz. Different sensory subtypes
contain different numbers of reconstructed neurons; equal per-neuron current
does not make their total input equal. No physiological stimulus-to-current
calibration is asserted. No shuffled-graph or biological-recording comparison
has been performed, so this does not establish overall biological validity.

## Evidence and analysis corrections

- `proprioceptor-circuit-001`: amplitude 0.2. Its legacy field
  `matches_circuit_prediction` means only relative preference. It is too broad
  a label and must not be read as a full circuit validation.
- `proprioceptor-circuit-002` and `003`: amplitudes 0.05 and 0.5. They separate
  relative preference and signed effects, but their SNpp50 flexor sign check
  oversimplifies the paper's time-dependent prediction. Those sign checks
  are not valid evidence about SNpp50's delayed disinhibition.
- `proprioceptor-circuit-004`: repeat at 0.2 with corrected sign reporting and
  runner checksum. SNpp50's flexor signs are deliberately not scored. Original
  records are preserved unchanged so the correction is auditable.

The repeat is not bitwise identical: comparing `001` and `004`, motor rates
differ by at most 1.4901161193847656e-08; motor-row IDs and timestamps match
exactly. An exact-equality reproducibility assertion therefore failed. This
small discrepancy is consistent with floating-point execution variability,
but its cause has not been isolated. The `004` runner checksum matches the
published script. All four neuromuscular unit tests pass independently.

## Consequence for the experiment

The initialized model is not devoid of structured sensorimotor responses.
However, this assay does not explain or repair its failure to generate useful
whole-body behavior. Before closing the muscle feedback loop, sensory tuning
polarity/gain and motor recruitment need explicit, testable hypotheses. Do not
choose them simply to force the desired movement, and do not mistake a spinal
reflex component for the goal of a richly behaving whole fly.
