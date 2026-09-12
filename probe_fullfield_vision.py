"""Full-field retinal diagnostic with intermediate CNS readouts; no body runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from model_loader import load_arrays


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','annotations','scene','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert sha(a.annotations)=='2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
    table=feather.read_table(a.annotations)
    table=table.filter(pc.fill_null(pc.equal(table['status'],'Traced'),False)).sort_by([('bodyId','ascending')])
    records=table.select(['bodyId','type']).to_pylist()
    sys.path.insert(0,str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays,_,digest=load_arrays(a.model)
    groups={'receptors':np.asarray(arrays['atlas.receptor_rows']).tolist(),
            'motor':np.asarray(arrays['atlas.motor_rows']).tolist()}
    for name in ('L1','L2','L3','Mi1','Tm3','T4','T5','DNa02','DNg13'):
        groups[name]=[i for i,r in enumerate(records) if r['type']==name or
                      (name in ('T4','T5') and (r['type'] or '').startswith(name))]
        assert groups[name],f'no annotated readout for {name}'
    rows=sorted({i for group in groups.values() for i in group});index={r:i for i,r in enumerate(rows)}
    scene=json.loads(a.scene.read_text());directions=np.asarray(scene['retinal_directions'])
    assert directions.shape==(1771,3)
    azimuth=np.arctan2(directions[:,0],-directions[:,2])
    conditions=['gray','bright','dark','positive-phase','negative-phase']
    torch.set_num_threads(8);device=torch.device('cuda:0')
    model=AnatomicalCNS(arrays,device=device).eval();state=model.initial_state(5)
    monitored=torch.as_tensor(rows,device=device);history=[];stimuli=[];began=time.monotonic()
    with torch.inference_mode():
        neutral=model.neutral_current(5).contiguous()
        body=model.body_mean[None].expand(5,-1).contiguous();context=torch.zeros((5,12),device=device)
        for tick in range(200):
            optic=np.full((5,1771,3),.5,dtype=np.float32)
            if 30<=tick<130:
                optic[1]=.95;optic[2]=.05
                for lane,sign in ((3,1),(4,-1)):
                    optic[lane]=(.5+.45*np.sin(8*azimuth+sign*4*np.pi*(tick*.01-.3)))[:,None]
            current=model.afferent_current(torch.as_tensor(optic,device=device),body,context).contiguous()
            state=model.step_from_current(current,state,neutral=neutral)
            assert all(torch.isfinite(v).all().item() for v in state.fields())
            history.append(state.rates[monitored].cpu().numpy().copy());stimuli.append(optic)
    rates=np.asarray(history);delta=rates[:,:,1:]-rates[:,:,:1];results=[]
    for lane,name in enumerate(conditions[1:]):
        summaries={}
        for group,ids in groups.items():
            values=delta[:,[index[i] for i in ids],lane]
            summaries[group]=dict(count=len(ids),peak_absolute_change=float(np.abs(values).max()),
                                 late_mean_absolute_change=float(np.abs(values[110:130]).mean()))
        results.append(dict(condition=name,groups=summaries))
    np.savez_compressed(a.output/'trace.npz',rates=rates,rows=np.asarray(rows),optic=np.asarray(stimuli),time=np.arange(1,201)*.01)
    report=dict(kind=__doc__,completed=True,runner_sha256=sha(Path(__file__)),model_manifest_sha256=digest,
        input_sha256={name:sha(getattr(a,name)) for name in ('annotations','scene')},
        conditions=conditions,groups={k:[dict(cns_row=i,**records[i]) for i in ids] for k,ids in groups.items()},
        results=results,wall_seconds=time.monotonic()-began,
        caveats=['Synthetic full-field input bypasses the physical ray sampler but uses the existing receptor projection.',
                 'Gratings use camera-local azimuth independently for each eye, not a calibrated world-relative optomotor stimulus.',
                 'Named readout groups are not asserted to form a single serial anatomical pathway.',
                 'No body, injected DN current, motor control, altered neural parameter or fitted decoder.',
                 'Response magnitudes are uncalibrated model rates, not physiological or behavioral validation.'])
    with (a.output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
