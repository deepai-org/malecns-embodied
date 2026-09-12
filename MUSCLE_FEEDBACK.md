# Physical muscle feedback, still not a whole fly

`run_muscle_feedback.py` closes the previously open muscle replay path:
MuJoCo foreleg angle → candidate claw-neuron current → full MaleCNS recurrence
→ named motor-neuron rates → six muscle activations → new physical angle.
There is no learned motor decoder, position-target controller or gait oscillator.
This component is not a replacement for the whole-body goal.

## Explicit assumptions

[Claw neurons encode joint position](https://elifesciences.org/articles/60299),
and [MANC annotations identify SNpp50/51 as claw subtypes](https://elifesciences.org/reviewed-preprints/97766v1).
Neither source establishes the particular tuning used here. We hypothesize two
opponent population currents, proportional to the physical tibia angle's
clipped fraction of its XML range. All cells in each subtype share tuning.
Both assignments of increasing/decreasing angle to SNpp50/51 are retained;
neither is selected on the basis of desirable motion. Peak current is 0.2 in
uncalibrated model units. This bypasses organ strain mechanics and is not a
validated sensory-organ model.

Six muscles use the existing cross-specimen anatomical-name mapping and equal
mean normalized motor rates. Nine unresolved muscles receive zero command;
minimum activation and passive mechanics remain. Other CNS inputs stay at the
model's neutral values. The body is fixed at its base, and only the left
foreleg has muscles. Gravity, passive mechanics and neural drive produce
initial-state motion; no external perturbation is applied in these trials.

Each polarity has three independent lanes: feedback, sensory current frozen
at its initial value, and zero muscle command with sensory feedback retained.
All bodies and neural states start identically. The frozen control distinguishes
physical feedback from tonic sensory input. The zero-motor control distinguishes
neural muscle drive from passive motion, not exact muscle paralysis.

## Evidence

`evidence/trials/muscle-feedback-001` ran all six lanes for 2 simulated seconds
in 17.76 wall seconds. For positive/reversed polarity respectively:

| Comparison | Positive | Reversed |
| --- | ---: | ---: |
| Feedback versus frozen sensory, max joint difference (rad) | 0.0008663 | 0.0006938 |
| Feedback versus frozen sensory, max motor-rate difference | 0.0011201 | 0.0010824 |
| Feedback versus zero motor, max joint difference (rad) | 0.531923 | 0.531343 |
| Maximum sensory current change from initial value | 0.0641491 | 0.0641489 |

The pathway has a measurable effect, but feedback contributes little compared
with tonic muscle drive and mechanical settling. This is not evidence of
walking, useful regulation, rich behavior or correct physiological tuning.

Trial `muscle-feedback-002` extended the same six conditions to 10 simulated
seconds in 87.13 wall seconds. All recorded values remained finite, with no
detected physics reset. Maximum feedback-versus-frozen joint differences stayed
below 0.000867 radians. During the final second, maximum within-joint ranges
were about 0.00068 radians in both driven conditions, versus 0.03558 in the
zero-motor conditions. The driven leg largely settles; this does not demonstrate
an endogenous locomotor rhythm. Both polarity hypotheses remain unresolved.

The original executed runner is preserved beside trial 001. Its only subsequent
change moved MuJoCo/Torch imports inside `main`, allowing sensory unit tests on
a machine without those packages. The initial local test attempt failed on
missing MuJoCo; after that change all six unit tests passed locally and on the
GPU host. Finite traces, identical physical initial states, the zero-motor and
frozen-sensory controls, the recorded angle-to-current transformation and the
trial 001 runner checksum were checked directly.
The same trace checks passed for trial 002, whose runner checksum matches the
current script. A local exact-equality check of muscle commands recomputed from
recorded motor rates failed on a 1.49e-08 difference. Cross-host floating-point
recomputation is not assumed bitwise identical; the explicit follow-up tolerance
is 6e-08 absolute, with zero clamps still checked exactly. Motor-row ordering
comes from the hashed-model-matched proprioceptor assay trace. Trial 001/002
are separate executions, not promised bitwise repeats.
The full follow-up check passed for every command in both trials; maximum
recomputation discrepancy was 2.9802322387695312e-08 in each trial.

## Reproduce

Use the source/model pins and installation in README.md, then:

```sh
.venv/bin/python run_muscle_feedback.py --source ../chreatures \
  --model ../releases/chreatures-v5-garden/model \
  --mapping muscle-probe-002.json --sensory proprioceptor-annotations.json \
  --xml ../FlyMimic/flymimic/assets/models/best_combined_cvt3.xml \
  --output runs/muscle-feedback --steps 200
```

Trace `qpos` contains the initial state plus post-step states, shaped
`[steps+1, lane, joint]`; `motor_rates` is `[steps, motor-neuron, lane]`;
`controls` and `sensory_current` are `[steps, lane, channel]`. Sensory currents
use the pre-step angle, motor rates follow that neural step, and muscle control
is applied for the following 0.01 physical seconds. See the receipt for lane
order, input hashes and assumptions. These finite runs do not install a service.
