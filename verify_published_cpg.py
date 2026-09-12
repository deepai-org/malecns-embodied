"""Recompute descriptive motor peak counts from a saved reference trace."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((a.trial/'result.json').read_text())
    assert r['completed'] and sha(a.trial/'trace.npz')==r['trace_sha256']
    with np.load(a.trial/'trace.npz',allow_pickle=False) as z:
        rates,t,ids,motors=z['rates'],z['time'],z['body_ids'].tolist(),z['motor_body_ids'].tolist()
    assert np.isfinite(rates).all() and rates.min()>=-1e-5 and rates.max()<=200.0001
    assert np.all(np.diff(t)>0) and len(rates)==3
    assert np.count_nonzero(rates[0])==0
    assert [c['stimulated_body_id'] for c in r['conditions']]==[None,10045,10056]
    late=t>=.5
    counts=[]
    for lane,c in enumerate(r['conditions']):
        assert [m['body_id'] for m in c['motor_metrics']]==motors
        count=0
        for m in c['motor_metrics']:
            v=rates[lane,ids.index(m['body_id']),late].astype(np.float64)
            peaks,_=find_peaks(v,prominence=1.)
            assert len(peaks)==m['peak_count']
            np.testing.assert_allclose(np.ptp(v),m['peak_to_peak'],rtol=0,atol=2e-5)
            np.testing.assert_allclose(v.mean(),m['mean_rate'],rtol=0,atol=2e-5)
            count+=len(peaks)>=3
        assert count==c['motors_with_at_least_3_peaks']
        counts.append(count)
    result=dict(verified=True,trial_receipt_sha256=sha(a.trial/'result.json'),
        verifier_sha256=sha(Path(__file__)),motor_peak_counts=counts,
        caveat='Trace reconstruction only; not solver-convergence proof, biological validation, or embodied behavior.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
