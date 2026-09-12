"""Physical screen and eye-ray input to full CNS, with body pose held fixed.

An open-loop sensory diagnostic, not an embodied behavioral demonstration.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from model_loader import load_arrays
from run_embodied import World, decode


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','scene','steering-receipt','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays,manifest,digest=load_arrays(a.model)
    scene=json.loads(a.scene.read_text())
    assert manifest['identity']['sensorySchema']==scene['cns_sensory_schema_sha256']
    reference=json.loads(a.steering_receipt.read_text());assert reference['model_manifest_sha256']==digest
    neurons=[]
    for c in reference['conditions'][1:]:
        if c['sign']==1:neurons.append(c['neuron'])
    assert len(neurons)==4
    conditions=['gray','static-grating','positive-phase','negative-phase']
    worlds=[];logs=[];optics=[];motors=[];dns=[];currents=[];began=time.monotonic()
    binary=a.source/'native/fly-world/target/release/chreatures-fly-world'
    try:
        for i in range(4):
            log=(a.output/f'world-{i}.stderr.log').open('w');logs.append(log)
            world=World(binary,a.scene,23,log);worlds.append(world)
            ready=world.receive();assert ready['residents']==4
        torch.set_num_threads(8);device=torch.device('cuda:0')
        model=AnatomicalCNS(arrays,device=device).eval();state=model.initial_state(16)
        motor_rows=torch.as_tensor(np.asarray(arrays['atlas.motor_rows'],dtype=np.int64),device=device)
        dn_rows=torch.as_tensor([n['cns_row'] for n in neurons],device=device)
        with torch.inference_mode():
            neutral=model.neutral_current(16).contiguous()
            body=model.body_mean[None].expand(16,-1).contiguous()
            context=torch.zeros((16,12),device=device)
            for tick in range(200):
                batch=[]
                for case,world in enumerate(worlds):
                    frame=np.full((64,128,3),.5,dtype=np.float32)
                    if 30<=tick<130 and case:
                        sign=(0,0,1,-1)[case]
                        phase=sign*2*(tick*.01-.3)
                        line=.5+.45*np.sin(2*np.pi*(4*np.arange(128)/128+phase))
                        frame[:]=line[None,:,None]
                    data=frame.astype('<f4').tobytes()
                    world.rpc('stimulus',screen=dict(width=128,height=64,
                        rgb_f32_base64=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest()),
                        sound=dict(position_mm=[0,0,0],frequency_hz=440,envelope=0,duration_s=.01))
                    packet=world.rpc('sample')['sample'];assert packet['time']==0
                    batch.append(decode(packet,'optic').reshape(4,1771,3))
                optic=np.concatenate(batch,axis=0);optics.append(optic)
                current=model.afferent_current(torch.as_tensor(optic,device=device),body,context).contiguous()
                currents.append((current-neutral).abs().max(0).values.cpu().numpy())
                state=model.step_from_current(current,state,neutral=neutral)
                assert all(torch.isfinite(v).all().item() for v in state.fields())
                motors.append(state.rates[motor_rows].cpu().numpy().copy())
                dns.append(state.rates[dn_rows].cpu().numpy().copy())
                if tick%50==0:print(json.dumps(dict(tick=tick)),flush=True)
        optic=np.asarray(optics);motor=np.asarray(motors);dn=np.asarray(dns)
        # Group by screen condition and identical resident identity.
        motor=motor.reshape(200,815,4,4);dn=dn.reshape(200,4,4,4)
        retinal=optic.reshape(200,4,4,1771,3)
        summary=[]
        for case in range(1,4):
            summary.append(dict(condition=conditions[case],
                peak_retinal_difference_from_gray=float(np.abs(retinal[:,case]-retinal[:,0]).max()),
                retinal_sites_changed_from_gray=int(np.any(np.abs(retinal[:,case]-retinal[:,0])>1e-7,axis=(0,1,3)).sum()),
                peak_motor_difference_from_gray=float(np.abs(motor[:,:,case]-motor[:,:,0]).max()),
                peak_dn_difference_from_gray=np.abs(dn[:,:,case]-dn[:,:,0]).max(axis=(0,2)).tolist(),
                peak_motor_difference_from_static=float(np.abs(motor[:,:,case]-motor[:,:,1]).max()),
                peak_dn_difference_from_static=np.abs(dn[:,:,case]-dn[:,:,1]).max(axis=(0,2)).tolist()))
        np.savez_compressed(a.output/'trace.npz',optic=retinal,motor_rates=motor,dn_rates=dn,
            time=np.arange(1,201)*.01,maximum_input_current_deviation=np.asarray(currents))
        report=dict(kind=__doc__,completed=True,runner_sha256=sha(Path(__file__)),
            model_manifest_sha256=digest,binary_sha256=sha(binary),scene_sha256=sha(a.scene),
            steering_receipt_sha256=sha(a.steering_receipt),world_client_sha256=sha(Path(__file__).with_name('run_embodied.py')),
            conditions=conditions,neurons=neurons,world_time=0,neural_seconds=2,
            screen=dict(width=128,height=64,cycles_across_width=4,cycles_per_second=2,mean=.5,amplitude=.45),
            receptor_channel_counts={str(k):int(v) for k,v in zip(*np.unique(np.asarray(arrays['graph.channel'])[np.asarray(arrays['atlas.receptor_rows'])],return_counts=True))},
            wall_seconds=time.monotonic()-began,results=summary,
            caveats=['Four separate world instances with identical initial seed/pose; physics is never advanced.',
                     'Retinal rays encounter the physical screen and scene, including occlusion; inputs are not painted directly onto neural rows.',
                     'Only optic input varies; BODY is at model mean and external context is zero.',
                     'Neural clock advances while body is fixed: prescribed visual presentation, not closed-loop behavior.',
                     'No DN injection, motor control, strength adjustment or parameter fitting.',
                     'Response differences are descriptive, not a calibrated visual behavior or physiology benchmark.'])
        with (a.output/'result.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(summary),flush=True)
    finally:
        for world in worlds:world.close()
        for log in logs:log.close()


if __name__=='__main__':main()
