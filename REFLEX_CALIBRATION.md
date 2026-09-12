# Physiological calibration: source registration before fitting

The disturbance assay's tiny neural contribution is not a target to repair by
optimizing recovery alone. We need distinct sensory, neural, and recruitment
constraints, with independent biological responses held out of fitting.

## Biological constraint

[Azevedo et al., Figure 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC7347388/)
measured tibia motor-neuron responses to imposed joint movement. Slow neurons'
firing is sensitive to small movements; fast/intermediate responses are often
subthreshold. Their response dynamics differ. Thus a pooled normalized rate
cannot be treated as either membrane voltage or calibrated firing frequency.
Matching movement alone would not establish the correct recruitment mechanism.

## Registered sources, not a completed calibration

`index_reflex_calibration.py` joins the authors' sensory-feedback cell table to
the public [Dryad dataset](https://doi.org/10.5061/dryad.76hdr7stb) metadata.
The saved index is `evidence/reflex-calibration-index-001.json`.

- FlyAnalysis revision: `c68159f1f7a4ecc957c708ae8411fd2550482f63`.
- Dataset version: `105831` (DOI `10.5061/dryad.76hdr7stb`).
- Indexed records: 15 fast, 15 intermediate, 14 slow.
- Applying the main membrane-potential plot's genotype exclusion leaves 7,
  11, and 10 respectively. **These are not the published figure sample sizes**;
  trial availability and further selection have not been reconstructed.

The exclusion follows the authors' script, removing iav-LexA genotypes and
81A07/ChR from that comparison. Commented-out table entries are not included.
No per-record traces have been loaded and no measured curve is synthesized
from the paper's prose. Dryad metadata succeeded, but the README API download
returned 401 and the public browser download returned 403 through both web and
curl. No authentication or access-control bypass was attempted. The public
author analysis checkout succeeded; its files were inspected, not executed.

The source table is MATLAB. Cell IDs identify physiological experiments, not
MaleCNS neurons. Our five left-foreleg Ti-flexor body IDs (807165, 809912, 818057,
819384, 909831) have the same coarse label in the committed annotation table;
none has yet been registered as the physiological fast/intermediate/slow cell.
Do not assign those classes by selecting whichever modeled response fits best.

## What the existing disturbance trace rules out

For each of those five cells, subtracting the yoked-sensory rate from the
feedback rate in `leg-disturbance-002` shows small individual responses. The
largest absolute difference across all four sign/polarity conditions is about
1.45e-5 normalized rate (body 807165); body 819384 stays below 1.50e-7.
Thus equal pooling is not concealing large opposing responses in this assay.
This does not establish that the upstream neural dynamics alone are defective:
sensory transduction and network operating state remain uncalibrated.

## Consequence for implementation

Do not promote a motor-gain change or a new recruitment law yet. First obtain
actual response records through an accessible authorized source and register
motor subclasses to anatomy, or use a physiological assay whose recorded cell
identity can already be matched. Fit observables in their correct units; keep
voltage, firing rate, and muscle force distinct. Split validation by recorded
cell/specimen, not by adjacent samples from the same recording. Maintain the
physical disturbance controls when reintegrating a candidate.

No parameters changed in this source-registration step. The full rich-fly goal
remains unmet, and access to this particular dataset is not a global project
blocker: anatomical registration and other published physiology remain usable
avenues.

```sh
python3 index_reflex_calibration.py --source ../FlyAnalysis \
  --output runs/reflex-calibration-index.json
```
