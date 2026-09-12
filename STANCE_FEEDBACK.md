# Full-CNS feedback from a clearance-checked stance

The whole-body runner now accepts an authenticated offline pose receipt with
`--initial-poses` and `--pose-index`. This changes **only initial joint/root
coordinates**. It does not hold the pose, add torque, initialize optimized
activations, change muscle strengths, or alter the CNS. Engine collision checks
reject starts with penetration above 1e-5 model units.

`six-leg-stance-feedback-001` uses candidate pose 1 from
`six-leg-stance-search-002.json`. Six independent physical worlds run for two
seconds with full MaleCNS on GPU0: feedback, frozen sensory input and zero motor
commands, each for both claw-sensory polarities. It took 125.95 wall seconds.
The original source keyframe remains the default when no pose is supplied.

**Result: no standing or useful behavior.** All six final conditions have
weight-bearing thorax and abdomen contacts. In the positive-polarity feedback
condition, reconstructed normal forces include thorax 2.467 and abdominal
segments A3–A6 3.519, 2.445, 3.649 and 1.285, in model force units. Nearly upright
body orientation is therefore not standing. Zero motor command still retains
passive muscle forces and minimum activation.

Late horizontal paths are 0.054–0.056 nominal mm for driven/frozen conditions,
versus 0.024 for zero commands; these are not classified as walking. Maximum
feedback/frozen joint differences are 0.107 and 0.239 radians for the two
polarities. Sensory feedback influences motion but is not demonstrated to help.

The known gross initial inter-leg overlap and free-fall start are removed in
this trial. Its negative outcome therefore does not depend on those starting
defects. It still cannot diagnose CNS competence independently of the unresolved
[muscle capacity/recruitment](SUPPORT_CAPACITY.md). Vision, odor, load sensing,
non-leg articulation and physiological calibration remain incomplete.

## Verification and provenance

- All six recorded initial poses exactly match the specified receipt entry.
- Recorded sensory transforms and named motor-to-muscle commands reconstruct;
  maximum command error is 4.48e-8. Twelve unmapped units remain zero-command.
- Thirty sampled 10-ms physical intervals restore from complete integration
  state and reproduce the next state exactly with recorded commands.
- Final contacts are reconstructed with `mj_forward` from complete saved states.
  Replay verifies physics, not independent CNS regeneration or biological fidelity.

Evidence: `evidence/trials/six-leg-stance-feedback-001/`,
`evidence/six-leg-stance-feedback-verification-001.json`, and
`evidence/six-leg-stance-state-replay-001.json`. The finite trial has ended; no
unattended experiment remains running. Historical runner hashes refer to the
executed git revision, not subsequently modified root scripts.

Reproduce by appending these options to the [six-leg run](SIX_LEG_BODY.md):

```sh
--initial-poses runs/refined-poses.json --pose-index 1
```

Verify the new trace using:

```sh
python3 verify_six_leg_feedback.py \
  --trial evidence/trials/six-leg-stance-feedback-001 \
  --initial-poses evidence/six-leg-stance-search-002.json \
  --output runs/stance-feedback-verification.json
```
