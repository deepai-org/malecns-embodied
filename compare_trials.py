"""Compare matched trajectories; never count passive settling as locomotion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('intact', type=Path)
parser.add_argument('sensory', type=Path)
parser.add_argument('motor', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
results, traces = {}, {}
for directory in (args.intact, args.sensory, args.motor):
    result = json.loads((directory / 'result.json').read_text())
    if not result['completed']:
        raise ValueError('incomplete trial')
    condition = result['condition']
    results[condition] = result
    with np.load(directory / 'trace.npz', allow_pickle=False) as archive:
        traces[condition] = {k: archive[k] for k in archive.files}
    result['trace_sha256'] = hashlib.sha256((directory / 'trace.npz').read_bytes()).hexdigest()
base = results['intact']
for result in results.values():
    for key in ('seed', 'steps', 'scene_sha256', 'model_manifest_sha256', 'binary_sha256'):
        if base[key] != result[key]:
            raise ValueError(f'unmatched trials: {key}')
summary = {}
for condition, trace in traces.items():
    reference = traces['intact']
    if not np.array_equal(reference['time'], trace['time']):
        raise ValueError('unmatched clocks')
    position = trace['thorax_position']
    # Discard the first second ONLY for this additional settling diagnostic.
    late = np.flatnonzero(trace['time'] >= 1.0)[0]
    summary[condition] = dict(
        total_xyz_displacement_mm=(position[-1]-position[0]).tolist(),
        after_one_second_xy_path_mm=np.linalg.norm(np.diff(position[late:,:,:2],axis=0),axis=-1).sum(axis=0).tolist(),
        max_root_difference_from_intact_mm=float(np.linalg.norm(position-reference['thorax_position'],axis=-1).max()),
        maximum_joint_command=float(np.abs(trace['motor'][:,:,:84]).max()),
        root_upright_min=float(trace['thorax_up'].min()),
        trace_sha256=results[condition]['trace_sha256'])
report = dict(trials=summary, caveats=[
    'Four residents share one world; these are not four independent replicates.',
    'Motor-disconnected clamps neural commands to zero; neutral-position servos remain powered.',
    'No biological or behavioral competence claim follows from execution or uprightness.',
])
with args.output.open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
