"""Compare saved CPG trajectories across integration tolerances."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','refined','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    receipts=[]; traces=[]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in (a.reference,a.refined):
        r=json.loads((folder/'result.json').read_text())
        assert r['completed'] and sha(folder/'trace.npz')==r['trace_sha256']
        receipts.append(r)
        with np.load(folder/'trace.npz',allow_pickle=False) as z:
            traces.append({k:z[k] for k in z.files})
    for key in ('source_revision','runner_sha256','input_sha256','parameters','max_step'):
        assert receipts[0][key]==receipts[1][key],key
    assert receipts[1]['rtol']<receipts[0]['rtol'] and receipts[1]['atol']<receipts[0]['atol']
    for key in ('time','body_ids','motor_body_ids'):
        np.testing.assert_array_equal(traces[0][key],traces[1][key])
    x,y=(t['rates'].astype(np.float64) for t in traces)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    indices=[traces[0]['body_ids'].tolist().index(i) for i in traces[0]['motor_body_ids']]
    results=[]
    for lane,(r,s) in enumerate(zip(receipts[0]['conditions'],receipts[1]['conditions'])):
        assert r['stimulated_body_id']==s['stimulated_body_id']
        delta=x[lane,indices]-y[lane,indices]
        results.append(dict(stimulated_body_id=r['stimulated_body_id'],
            motor_rate_max_abs_difference=float(np.abs(delta).max()),
            motor_rate_rms_difference=float(np.sqrt(np.mean(delta**2))),
            reference_motor_peak_counts=[m['peak_count'] for m in r['motor_metrics']],
            refined_motor_peak_counts=[m['peak_count'] for m in s['motor_metrics']],
            reference_repeated_peak_motors=r['motors_with_at_least_3_peaks'],
            refined_repeated_peak_motors=s['motors_with_at_least_3_peaks']))
    out=dict(completed=True,receipt_sha256=[sha(f/'result.json') for f in (a.reference,a.refined)],
        tolerances=[dict(rtol=r['rtol'],atol=r['atol']) for r in receipts],conditions=results,
        all_recorded_rate_max_abs_difference=float(np.abs(x-y).max()),
        caveat='Two-tolerance empirical comparison; no universal convergence or biological-validity claim.')
    with a.output.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({**out,'conditions':[{k:v for k,v in r.items() if not k.endswith('counts')} for r in results]},indent=2))


if __name__=='__main__':main()
