"""Full-CNS steering-neuron pulse assay; no body or runtime DN controller."""
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


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','annotations','motor','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--amplitude',type=float,default=.2)
    a=p.parse_args()
    assert 0<a.amplitude<=1
    a.output.mkdir(parents=True,exist_ok=False)
    annotation_sha=sha(a.annotations)
    assert annotation_sha=='2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
    table=feather.read_table(a.annotations)
    table=table.filter(pc.fill_null(pc.equal(table['status'],'Traced'),False)).sort_by([('bodyId','ascending')])
    fields=['bodyId','type','instance','somaSide','rootSide','superclass','subclass','mancType']
    selected=[];interneurons=[]
    for row,r in enumerate(table.select(fields).to_pylist()):
        if r['type'] in ('DNa02','DNg13'):selected.append(dict(cns_row=row,**r))
        if r['type'] in ('IN19A003','IN08A006'):interneurons.append(dict(cns_row=row,**r))
    selected.sort(key=lambda r:(r['type'],r['somaSide']))
    assert len(selected)==4 and all(r['superclass']=='descending_neuron' for r in selected)
    assert len({(r['type'],r['somaSide']) for r in selected})==4
    motor=json.loads(a.motor.read_text());assert motor['source_sha256']==annotation_sha
    sys.path.insert(0,str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays,_,digest=load_arrays(a.model)
    motor_rows=np.asarray(arrays['atlas.motor_rows'],dtype=np.int64)
    motor_index={int(r):i for i,r in enumerate(motor_rows)}
    assert len(table)==165122 and all(r['cns_row'] not in motor_index for r in selected)
    groups=[]
    for side in ('L','R'):
        for leg in ('fl','ml','hl'):
            records=[r for r in motor['motors'] if r['subclass']==leg and (r['somaSide'] or r['rootSide'])==side]
            groups.append(dict(side=side,leg=leg,indices=[motor_index[r['cns_row']] for r in records]))
    conditions=[dict(kind='control',sign=0,neuron=None)]
    conditions += [dict(kind='pulse',sign=sign,neuron=r) for r in selected for sign in (1,-1)]
    torch.set_num_threads(8)
    device=torch.device('cuda:0');model=AnatomicalCNS(arrays,device=device).eval()
    state=model.initial_state(len(conditions))
    monitored=[r['cns_row'] for r in selected+interneurons]
    rows=torch.as_tensor(motor_rows,device=device);extra=torch.as_tensor(monitored,device=device)
    motors=[];observed=[];began=time.monotonic()
    with torch.inference_mode():
        neutral=model.neutral_current(len(conditions)).contiguous()
        pulse=torch.zeros_like(neutral)
        for lane,c in enumerate(conditions[1:],1):pulse[c['neuron']['cns_row'],lane]=a.amplitude*c['sign']
        for tick in range(200):
            current=neutral+pulse if 30<=tick<130 else neutral
            state=model.step_from_current(current.contiguous(),state,neutral=neutral)
            assert all(torch.isfinite(v).all().item() for v in state.fields())
            motors.append(state.rates[rows].cpu().numpy().copy())
            observed.append(state.rates[extra].cpu().numpy().copy())
    rates=np.asarray(motors);monitored_rates=np.asarray(observed)
    delta=rates[:,:,1:]-rates[:,:,:1]
    results=[]
    for lane,c in enumerate(conditions[1:]):
        late=delta[110:130,:,lane].mean(axis=0)
        summaries=[]
        for g in groups:
            values=late[g['indices']]
            summaries.append(dict(side=g['side'],leg=g['leg'],mean_change=float(values.mean()),
                mean_absolute_change=float(np.abs(values).mean()),maximum_absolute_change=float(np.abs(values).max())))
        side=c['neuron']['somaSide']
        ipsi=float(np.mean([g['mean_absolute_change'] for g in summaries if g['side']==side]))
        contra=float(np.mean([g['mean_absolute_change'] for g in summaries if g['side']!=side]))
        results.append(dict(condition=c,peak_motor_change=float(np.abs(delta[:,:,lane]).max()),
            late_leg_groups=summaries,ipsilateral_mean_absolute_change=ipsi,contralateral_mean_absolute_change=contra,
            ipsilateral_fraction=None if ipsi+contra==0 else ipsi/(ipsi+contra)))
    np.savez_compressed(a.output/'trace.npz',time=np.arange(1,201)*.01,
        motor_rates=rates,motor_rows=motor_rows,monitored_rates=monitored_rates,monitored_rows=np.asarray(monitored))
    report=dict(kind=__doc__,completed=True,runner_sha256=sha(Path(__file__)),
        model_manifest_sha256=digest,annotation_sha256=annotation_sha,motor_sha256=sha(a.motor),
        conditions=conditions,interneurons=interneurons,motor_groups=groups,amplitude=a.amplitude,
        stimulation_interval=[.3,1.3],late_readout='last 20 stimulated updates',
        wall_seconds=time.monotonic()-began,results=results,
        references=['https://pubmed.ncbi.nlm.nih.gov/39293446/',
                    'https://elifesciences.org/articles/96084'],
        caveats=['Direct diagnostic DN current, not physical sensory transduction or an installed behavior controller.',
                 'Experimental DNa02/DNg13 effects concern leg kinematics, not mean motor-rate magnitudes.',
                 'Side-bias summaries are exploratory neural diagnostics, not a biological pass/fail score.',
                 'Model current/rates are not calibrated to injected picoamps or recorded hertz.',
                 'Neutral resting context is not the walking state used in biological perturbations.',
                 'No body, muscles, neural decoder, learned policy or parameter fitting runs in this assay.'])
    with (a.output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([dict(type=r['condition']['neuron']['type'],side=r['condition']['neuron']['somaSide'],
        sign=r['condition']['sign'],peak=r['peak_motor_change'],ipsilateral_fraction=r['ipsilateral_fraction']) for r in results]),flush=True)


if __name__=='__main__':main()
