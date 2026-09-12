# Attribution and provenance

Experiment code in this repository is distributed under AGPL-3.0-or-later.

The neural and whole-body baselines execute **Chreatures**, by emberian and
contributors, under AGPL-3.0-or-later. Its source is pinned but not vendored here.
The model/physics interface and array shapes in the experiment scripts follow
that project's published contracts. Preserve its own notices when downloading
or distributing its artifacts.

The muscle assets come from **FlyMimic**, by Pembe Gizem Özdil, Chuanfang Ning,
and collaborators, under Apache-2.0. Cite *Musculoskeletal simulation of limb
movement biomechanics in Drosophila melanogaster*, ICLR 2026,
https://arxiv.org/abs/2509.06426. Those assets are not vendored here.

`motor-annotations.json` and `proprioceptor-annotations.json` are subsets derived
from the **MaleCNS v1.0** official body annotations, distributed under CC BY 4.0:
https://male-cns.janelia.org/download/ and
https://creativecommons.org/licenses/by/4.0/ . The export scripts select Traced
neurons, sort by bodyId, select the indicated classes, and add local CNS row
indices. These modifications do not create new biological annotations.

Physical bodies derive from NeuroMechFly/FlyGym; Chreatures explicitly combines
a female-derived morphology with a different male CNS specimen. MuJoCo supplies
physics. Consult upstream releases for their individual asset/software notices.

Evidence JSON/NPZ records are outputs of this experiment. Their presence is not
an endorsement by upstream authors, proof of biological validity, or a claim
that a living fly has been emulated. The repository preserves negative results.
