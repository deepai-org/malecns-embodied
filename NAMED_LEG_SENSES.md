# Named whole-body leg-position feedback

`run_named_senses.py` now closes the whole-body physical loop through 91 named
SNpp50/51 claw afferents, replacing the broad random body projection **only on
those rows**. It retains full MaleCNS dynamics, the existing aggregate muscles,
physics-step force updates, and all other sensory inputs. There is no gait
oscillator, task signal, motor decoder or direct angle-to-motor bypass.

Annotations identify eleven subtype/leg groups spanning six legs. No right-front
SNpp50 group is present in the selected traced annotations; it is recorded as
missing, not synthesized. Runtime checks require all selected cells to belong
to the authenticated body-afferent set and exclude motor neurons. Per-group
body IDs, row indices and physical addresses are recorded in every trial receipt.

## Evidence versus approximation

Claw afferents encode joint position; the anatomy distinguishes claw from
movement-sensitive hook and vibration-sensitive club populations. Sources:
[MANC circuit study, Figure 59](https://elifesciences.org/reviewed-preprints/97766v1),
[proprioceptor circuit study](https://www.sciencedirect.com/science/article/pii/S0960982221012756),
and [primary BANC annotation for SNpp50](https://www.virtualflybrain.org/blog/2022/01/01/banc_626720575941507004194-vfb_00105yf0/).
These sources do not calibrate our MaleCNS-to-rig coordinate transform.

The implemented transduction is explicitly approximate:

```text
current = model_neutral_current + 0.2 * polarity * subtype_sign * tanh(q - rig_rest)
```

SNpp50/51 use opposite signs; both global polarity hypotheses are tested. The
one-radian tuning scale, amplitude, neutral center, homogeneous within-subtype
tuning and serial leg correspondence are assumptions. This is joint-position
transduction, not modeled chordotonal tendon strain or measured firing rates.
No polarity is selected based on motion or uprightness. Unidentified sensory
rows still use the upstream broad interface; the full sensory system is not
claimed to be anatomically reconstructed.

## Bounded whole-body results

Four separate two-second runs completed on the four GPUs, each containing four
residents sharing a physical world. All use the no-joint-passives scene. Four
residents are not independent replicates.

| Condition | Maximum named antagonist rate separation | Observed behavior |
| --- | --- | --- |
| Positive sensory polarity | 0.000520 | Tipping/settling; no useful locomotion demonstrated |
| Negative sensory polarity | 0.000644 | Tipping/settling; no useful locomotion demonstrated |
| Frozen senses, positive polarity | 0.000104 | Tipping/settling |
| Zero muscle drive, positive polarity | 0.00764 | Passive motion; neural output cannot actuate body |

Zero-drive neural activity is larger despite identical CNS parameters: its
different physical trajectory generates different senses. This reinforces why
near-tonic rates in one embodied condition do not prove intrinsic CNS incapacity.
It does not identify which sensory population caused that response.

The positive-polarity versus frozen root trajectories differ by up to 0.130 mm.
The frozen control holds **all** senses, including vision and the remaining broad
body projection, so that comparison does not isolate the named adapter's effect.
There is no evidence of beneficial coordination or corrected biological reflexes.

`verify_named_senses.py` reconstructed all named sensory offsets and muscle
coefficients exactly from the numeric traces. Unsupported actuator forces were
zero, and zero-drive qpos exactly matched the prior physics-step zero-drive run.
All 16 Python unit tests passed. These checks establish implementation integrity,
not biological fidelity or behavioral competence.

## Reproduce

Use the existing physics-step host and authenticated no-passives scene:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_named_senses.py \
  --source ../chreatures-muscle-step \
  --model ../releases/chreatures-v5-garden/model \
  --scene ../scenes/no-joint-passives/world.json \
  --motor motor-annotations.json --sensory proprioceptor-annotations.json \
  --sensory-polarity 1 --output runs/named-plus --steps 200
```

Repeat into fresh paths with `--sensory-polarity -1`, or
`--condition frozen-sensory` / `--condition zero-motor`. No persistent background
process is installed. Original runners, neural parameters and source pins remain
unchanged. Receipts and numeric traces: `evidence/trials/named-senses-*-001`.
Verification: `evidence/named-senses-verification-001.json`.

This replaces a small but identifiable part of the sensory interface. The next
major unresolved interface is recruitment/force scaling and incomplete leg
actuation; more variants of this position curve are not the priority. The full
objective remains rich emergent whole-fly behavior, not this component milestone.
