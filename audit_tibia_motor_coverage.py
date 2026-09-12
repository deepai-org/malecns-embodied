"""Audit named tibia motor coverage and individual disturbance responses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('trial','motor','mapping','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a = p.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((a.trial/'result.json').read_text())
    assert receipt['completed'] and sha(a.trial/'trace.npz') == receipt['trace_sha256']
    mapping = json.loads(a.mapping.read_text())
    assert sha(a.mapping) == receipt['input_sha256']['mapping']
    annotation = json.loads(a.motor.read_text())
    assert annotation['source_sha256'] == mapping['neuron_source_sha256']
    with np.load(a.trial/'trace.npz',allow_pickle=False) as z:
        rates,rows = z['motor_rates'],z['motor_rows'].tolist()
    result = []
    for neuron in annotation['motors']:
        if neuron['subclass'] != 'fl' or (neuron['somaSide'] or neuron['rootSide']) != 'L':
            continue
        if neuron['mancType'] not in ('Ti flexor MN','Acc. ti flexor MN','Ti extensor MN'):
            continue
        idx = rows.index(neuron['cns_row'])
        changes = []
        for base in range(0,12,3):
            conditions = receipt['conditions'][base:base+3]
            assert [c['mode'] for c in conditions] == ['sham','feedback','yoked-sensory']
            delta = rates[:,idx,base+1]-rates[:,idx,base+2]
            changes.append(dict(polarity=conditions[0]['polarity'],torque_sign=conditions[0]['sign'],
                min_rate_change=float(delta.min()),max_rate_change=float(delta.max())))
        result.append(dict(body_id=neuron['bodyId'],cns_row=neuron['cns_row'],label=neuron['mancType'],
            mapped_muscles=[m['muscle'] for m in mapping['muscles'] if neuron['cns_row'] in m['candidate_motor_rows']],
            physiological_speed_class=None,responses=changes))
    assert len(result) == 17
    out = dict(kind='named left-foreleg tibia motor coverage; not physiological registration',
        source_sha256={n:sha(getattr(a,n)) for n in ('motor','mapping')},
        trial_receipt_sha256=sha(a.trial/'result.json'),runner_sha256=sha(Path(__file__)),neurons=result,
        limitations=['Accessory-flexor pool has no actuator in this body; no anatomical assignment invented.',
            'Main/accessory labels do not establish individual fast/intermediate/slow identities.',
            'Normalized model responses cannot be compared numerically to voltage or Hz without calibration.'])
    with a.output.open('x') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(dict(neurons=len(result),unmapped=sum(not r['mapped_muscles'] for r in result))))


if __name__ == '__main__':
    main()
