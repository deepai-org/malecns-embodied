"""Verify saved steering-assay identities and recompute descriptive metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent
    motor=json.loads((root/'motor-annotations.json').read_text())['motors']
    reports=[]
    for trial in a.trials:
        r=json.loads((trial/'result.json').read_text())
        assert r['completed'] and r['runner_sha256']==sha(root/'probe_steering_circuit.py')
        assert r['motor_sha256']==sha(root/'motor-annotations.json')
        assert r['annotation_sha256']==json.loads((root/'motor-annotations.json').read_text())['source_sha256']
        with np.load(trial/'trace.npz',allow_pickle=False) as t:
            rates=t['motor_rates'].copy();rows=t['motor_rows'].copy()
            observed=t['monitored_rates'].copy();monitored=t['monitored_rows'].copy()
            np.testing.assert_allclose(t['time'],np.arange(1,201)*.01,rtol=0,atol=1e-12)
        assert rates.shape==(200,815,9) and np.isfinite(rates).all() and np.isfinite(observed).all()
        assert set(rows)=={m['cns_row'] for m in motor}
        assert all(c['neuron']['cns_row'] in monitored and c['neuron']['cns_row'] not in rows for c in r['conditions'][1:])
        for g in r['motor_groups']:
            expected={m['cns_row'] for m in motor if m['subclass']==g['leg'] and (m['somaSide'] or m['rootSide'])==g['side']}
            assert set(rows[g['indices']])==expected
        delta=rates[:,:,1:]-rates[:,:,:1]
        prestimulus=float(np.abs(delta[:30]).max());assert prestimulus<1e-7
        cases=[]
        for lane,case in enumerate(r['results']):
            late=delta[110:130,:,lane].mean(0)
            side=case['condition']['neuron']['somaSide']
            ipsi=[];contra=[]
            for group,recorded in zip(r['motor_groups'],case['late_leg_groups']):
                values=late[group['indices']]
                value=float(np.abs(values).mean())
                np.testing.assert_allclose(value,recorded['mean_absolute_change'],rtol=0,atol=1e-12)
                (ipsi if group['side']==side else contra).append(value)
            fraction=float(np.mean(ipsi)/(np.mean(ipsi)+np.mean(contra)))
            np.testing.assert_allclose(fraction,case['ipsilateral_fraction'],rtol=0,atol=1e-12)
            target=int(np.flatnonzero(monitored==case['condition']['neuron']['cns_row'])[0])
            target_change=float((observed[110:130,target,lane+1]-observed[110:130,target,0]).mean())
            assert target_change*case['condition']['sign']>0
            cases.append(dict(type=case['condition']['neuron']['type'],side=side,sign=case['condition']['sign'],
                target_rate_change=target_change,ipsilateral_fraction=fraction,
                peak_motor_change=float(np.abs(delta[:,:,lane]).max())))
        reports.append(dict(trial=trial.name,amplitude=r['amplitude'],trace_sha256=sha(trial/'trace.npz'),
            maximum_prestimulus_difference=prestimulus,cases=cases))
    result=dict(checked=True,verifier_sha256=sha(Path(__file__)),reports=reports,
        caveat='Reconstructs descriptive metrics and input identities, not independent neural recurrence or behavioral validity.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
