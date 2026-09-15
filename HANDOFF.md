# Stopping point — 2026-09-12

## Infrastructure closed — 2026-09-15

At the user's explicit request, the specific four-RTX-3090 Vast instance was
destroyed. Before deletion, its address/SSH mapping and saved trial checksum
were matched to this experiment. Vast returned success and a subsequent
account listing no longer contained the instance; other instance IDs were
unchanged. Its remote disk is no longer available. Committed source, evidence
and local files remain; a future restart requires a new host and environment.
The provisioning statements below describe the earlier September 12 handoff.

Paused at the user's request. The full objective remains unfinished:
physical environment → sensory organs → MaleCNS dynamics → motor neurons →
muscles/body → changed sensory input, producing rich fly behavior.

## Current state

- Full curated MaleCNS runs, but the integrated six-leg muscle body rests on its
  thorax/abdomen. Useful standing, walking, grooming, feeding and flight are not
  demonstrated. See `STANCE_FEEDBACK.md`.
- Physical leg disturbances recover almost identically with sham-yoked sensory
  control: feedback changes recovery error by less than 0.003%. This is not a
  useful corrective reflex. See `LEG_DISTURBANCE.md`.
- The independent published mCNS subnetwork reference generates repeated motor
  peaks under constant DNg100 input, with a silent zero-input control. Counts
  (8 and 19) survive tighter tolerances; trajectories agree approximately, not
  exactly. See `PUBLISHED_CPG.md` and trials `published-cpg-003/004`.
- The reference is a 4,310-cell subnetwork, not the full artifact. A five-synapse
  threshold explains much of its connectivity difference, but ROI scope,
  transmitter selection and remaining differences need explicit treatment.
- No reference dynamics, new gain, threshold filter or oscillator has been
  installed in the embodied fly.

## Operational handoff

The Vast host was reachable at handoff. Read-only process inspection found no
running experiment; all four GPUs reported 0% utilization and 1 MiB used each.
No new background service or scheduled experiment was installed. The machine
remains provisioned; it was not shut down or destroyed.

Local worktree: `/home/ubuntu/malecns-embodied`.
Remote experiment root: `/opt/malecns`.
Private connection details and credentials are deliberately absent from this
public repository. Existing source/data pins and trial outputs were preserved.
Unrelated user work in other repositories was not modified.

The completed threshold audit and older motion-asymmetry analysis are now
published with their receipts. Both script hashes match the executed versions.
The interrupted, unexecuted signed-weight extension was removed; no result is
claimed for it. Historical failed launches are described in their experiment
notes, and their output directories were not reused.

## Recommended next decisive experiment

Keep the published rhythm reference as a positive control. Build a separately
versioned full-MaleCNS candidate using its dynamics, with graph selection and
transmitter handling explicit and separately testable. Test sustained,
structured motor activity and silent/lesioned controls before connecting it to
muscles. Do not insert the extracted circuit as an external gait generator.

If the full network passes, integrate it with calibrated recruitment and a body
shown mechanically capable of support. Then require sensory-driven initiation,
coordination, orientation and recovery in the same physical loop. Those remain
milestones, not a redefinition of success around walking alone.

Avoid another open-ended series of small diagnostic additions. The next phase
should test a concrete candidate against the present baseline and produce an
accept/reject decision. Rich embodied behavior still requires substantive
model development; there is no defensible completion date or success probability.
