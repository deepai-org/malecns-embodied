"""Recompute saved visual-group summaries; no independent CNS regeneration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.trial/'result.json').read_text())
    assert r['completed'] and r['runner_sha256']==sha(Path(__file__).with_name('probe_fullfield_vision.py'))
    with np.load(a.trial/'trace.npz',allow_pickle=False) as t:
        rates=t['rates'];rows=t['rows'];optic=t['optic']
        assert rates.shape==(200,len(rows),5) and optic.shape==(200,5,1771,3)
        assert np.isfinite(rates).all() and np.isfinite(optic).all()
        assert np.all((optic>=0)&(optic<=1))
        np.testing.assert_allclose(t['time'],np.arange(1,201)*.01,rtol=0,atol=1e-12)
        index={int(v):i for i,v in enumerate(rows)}
        assert len(index)==len(rows)
        delta=rates[:,:,1:]-rates[:,:,:1]
        before=float(np.abs(delta[:30]).max());assert before==0
        checked=0
        for lane,result in enumerate(r['results']):
            for name,group in r['groups'].items():
                values=delta[:,[index[n['cns_row']] for n in group],lane]
                assert len(group)==result['groups'][name]['count']
                assert float(np.abs(values).max())==result['groups'][name]['peak_absolute_change']
                assert float(np.abs(values[110:130]).mean())==result['groups'][name]['late_mean_absolute_change']
                checked+=1
    report=dict(checked=True,group_summaries=checked,maximum_prestimulus_difference=before,
        verifier_sha256=sha(Path(__file__)),trace_sha256=sha(a.trial/'trace.npz'),
        caveat='Reconstructs recorded group metrics, not independent neural recurrence, anatomical pathways or behavior.')
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
