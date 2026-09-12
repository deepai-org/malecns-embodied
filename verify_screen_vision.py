"""Reconstruct screen-assay contrasts; not a behavioral validator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent
    r=json.loads((a.trial/'result.json').read_text())
    assert r['completed'] and r['world_time']==0 and r['neural_seconds']==2
    assert r['runner_sha256']==sha(root/'probe_screen_vision.py')
    assert r['world_client_sha256']==sha(root/'run_embodied.py')
    ref=root/'evidence/trials/steering-circuit-001/result.json'
    assert r['steering_receipt_sha256']==sha(ref)
    assert r['model_manifest_sha256']==json.loads(ref.read_text())['model_manifest_sha256']
    with np.load(a.trial/'trace.npz',allow_pickle=False) as a_np:
        arrays={k:a_np[k].copy() for k in a_np.files}
    assert all(np.isfinite(x).all() for x in arrays.values())
    optic,motor,dn=(arrays[k] for k in ('optic','motor_rates','dn_rates'))
    assert optic.shape==(200,4,4,1771,3) and motor.shape==(200,815,4,4) and dn.shape==(200,4,4,4)
    assert np.all((optic>=0)&(optic<=1))
    np.testing.assert_allclose(arrays['time'],np.arange(1,201)*.01,rtol=0,atol=1e-12)
    pre=[]
    for case,result in enumerate(r['results'],1):
        differences=dict(peak_retinal_difference_from_gray=float(np.abs(optic[:,case]-optic[:,0]).max()),
            peak_motor_difference_from_gray=float(np.abs(motor[:,:,case]-motor[:,:,0]).max()),
            peak_dn_difference_from_gray=np.abs(dn[:,:,case]-dn[:,:,0]).max(axis=(0,2)),
            peak_motor_difference_from_static=float(np.abs(motor[:,:,case]-motor[:,:,1]).max()),
            peak_dn_difference_from_static=np.abs(dn[:,:,case]-dn[:,:,1]).max(axis=(0,2)))
        for name,value in differences.items():np.testing.assert_allclose(value,result[name],rtol=0,atol=1e-12)
        before=max(float(np.abs(optic[:30,case]-optic[:30,0]).max()),
                   float(np.abs(motor[:30,:,case]-motor[:30,:,0]).max()),
                   float(np.abs(dn[:30,:,case]-dn[:30,:,0]).max()))
        assert before<1e-7;pre.append(before)
    result=dict(checked=True,verifier_sha256=sha(Path(__file__)),trace_sha256=sha(a.trial/'trace.npz'),
        maximum_prestimulus_contrast=max(pre),
        caveat='Reconstructs saved contrasts and matching before stimulation; not independent physics/neural replay or visual behavior.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
