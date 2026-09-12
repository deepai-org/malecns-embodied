"""Full MaleCNS proprioceptive loop with the replicated six-leg muscle body.

No vision/odor transduction yet in this alternative body. Not rich behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import torch
from model_loader import load_arrays
from pooled_muscle_mapping import POOLS
from run_pooled_feedback import position_current


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','xml','body-receipt','motor','sensory','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--initial-poses',type=Path)
    p.add_argument('--pose-index',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.steps<=0:raise ValueError('steps must be positive')
    sys.path.insert(0,str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays,_,digest=load_arrays(a.model)
    body_receipt=json.loads(a.body_receipt.read_text())
    assert sha(a.xml)==body_receipt['xml_sha256']
    motor=json.loads(a.motor.read_text());sensory=json.loads(a.sensory.read_text())
    assert motor['source_sha256']==sensory['source_sha256']
    physics=mujoco.MjModel.from_xml_path(str(a.xml))
    assert physics.nu==physics.na==90 and physics.nv==48 and physics.neq==0
    assert np.count_nonzero(physics.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)==1
    substeps=round(.01/physics.opt.timestep)
    assert abs(substeps*physics.opt.timestep-.01)<1e-12
    motor_rows=np.asarray(arrays['atlas.motor_rows'],dtype=np.int64);index={int(r):i for i,r in enumerate(motor_rows)}
    weights=np.zeros((90,len(motor_rows)),dtype=np.float32);recruitment=[]
    for i in range(90):
        tag,name=physics.actuator(i).name.split('/',1)
        labels=POOLS[name];side=tag[0].upper();subclass={'f':'fl','m':'ml','h':'hl'}[tag[1]]
        selected=[r for r in motor['motors'] if r['mancType'] in labels and r['subclass']==subclass and (r['somaSide'] or r['rootSide'])==side]
        rows=[r['cns_row'] for r in selected]
        if rows:weights[i,[index[r] for r in rows]]=1/len(rows)
        recruitment.append(dict(actuator=physics.actuator(i).name,rows=rows,body_ids=[r['bodyId'] for r in selected],
            missing_pool_labels=sorted(set(labels)-{r['mancType'] for r in selected})))
    senses=[];body_rows=set(map(int,arrays['atlas.body_rows']))
    for tag in ('lf','lm','lh','rf','rm','rh'):
        joint=physics.joint(tag+'/joint_LFTibia_pitch')
        groups=[]
        for kind in ('SNpp50','SNpp51'):
            nerve={'f':'ProLN','m':'MesoLN','h':'MetaLN'}[tag[1]]
            selected=[r for r in sensory['neurons'] if r['type']==kind and r['entryNerve']==nerve and (r['somaSide'] or r['rootSide'])==tag[0].upper()]
            rows=[r['cns_row'] for r in selected]
            if any(r not in body_rows or r in index for r in rows):raise ValueError('invalid sensory boundary')
            groups.append(rows)
        senses.append(dict(leg=tag,qpos=int(joint.qposadr[0]),limits=joint.range.tolist(),groups=groups))
    conditions=[(mode,polarity) for polarity in (1,-1) for mode in ('feedback','frozen-sensory','zero-motor')]
    bodies=[mujoco.MjData(physics) for _ in conditions]
    for b in bodies:mujoco.mj_resetDataKeyframe(physics,b,0);mujoco.mj_forward(physics,b)
    initialization=None
    if a.initial_poses:
        poses=json.loads(a.initial_poses.read_text())
        assert poses['xml_sha256']==sha(a.xml)
        if not 0<=a.pose_index<len(poses['results']):raise ValueError('invalid pose index')
        pose=poses['results'][a.pose_index]
        assert pose['geometric_candidate']
        for b in bodies:
            b.qpos[:]=pose['qpos'];b.qvel[:]=0;mujoco.mj_forward(physics,b)
            assert max((-float(c.dist) for c in b.contact),default=0.)<1e-5
        initialization=dict(poses_sha256=sha(a.initial_poses),pose_index=a.pose_index,
            use='Initial qpos only; no ongoing optimizer, target pose or corrective force')
    initial=bodies[0].qpos.copy();thorax=physics.body('Thorax').id
    torch.set_num_threads(8);cns=AnatomicalCNS(arrays,device=torch.device('cuda:0')).eval()
    state=cns.initial_state(6);neural_rows=torch.as_tensor(motor_rows,device='cuda:0')
    identity=dict(kind='free six-leg muscle body with full-CNS proprioception; not rich behavior',
        model_manifest_sha256=digest,runner_sha256=sha(Path(__file__)),
        input_sha256={n:sha(getattr(a,n)) for n in ('xml','body_receipt','motor','sensory')},
        pooling_code_sha256=sha(Path(__file__).with_name('pooled_muscle_mapping.py')),
        sensory_code_sha256=sha(Path(__file__).with_name('run_pooled_feedback.py')),
        conditions=conditions,recruitment=recruitment,senses=senses,steps=a.steps,
        initialization=initialization,
        assumptions=body_receipt['assumptions']+[
            'Named pool sharing extended by leg and side; absent pools get zero command, partial composite pools are recorded.',
            'Only named claw position input varies; all other sensory currents remain at model neutral values.',
            'Position transduction and normalized-rate recruitment are not physiological calibration.',
            'Zero command is not exact paralysis: source passive muscle forces and minimum activation remain.',
            'Six conditions use separate physical worlds; no learned decoder, trajectory or behavior selector.'])
    (a.output/'intent.json').write_text(json.dumps(identity,indent=2))
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state_size=mujoco.mj_stateSize(physics,state_spec)
    identity['physics_state_spec']=int(state_spec);identity['physics_state_size']=state_size
    # Rewrite our newly created intent before any step to include the replay contract.
    (a.output/'intent.json').write_text(json.dumps(identity,indent=2))
    history={k:[] for k in ('time','qpos','qvel','act','integration_state','controls','motor_rates','sensory_current','actuator_force','thorax_position','thorax_up')}
    began=time.monotonic()
    try:
        with torch.inference_mode():
            neutral=cns.neutral_current(6).contiguous()
            for tick in range(a.steps+1):
                history['time'].append(bodies[0].time)
                for key in ('qpos','qvel','act','actuator_force'):history[key].append(np.stack([getattr(b,key).copy() for b in bodies]))
                snapshots=[]
                for b in bodies:
                    snapshot=np.empty(state_size);mujoco.mj_getState(physics,b,snapshot,state_spec);snapshots.append(snapshot)
                    mujoco.mj_kinematics(physics,b)
                history['integration_state'].append(np.stack(snapshots))
                history['thorax_position'].append(np.stack([b.xpos[thorax].copy() for b in bodies]))
                history['thorax_up'].append([float(b.xmat[thorax].reshape(3,3)[2,2]) for b in bodies])
                if tick==a.steps:break
                current=neutral.clone();values=[]
                for lane,((mode,polarity),b) in enumerate(zip(conditions,bodies)):
                    q=initial if mode=='frozen-sensory' else b.qpos;sample=[]
                    for s in senses:
                        v=position_current(q[s['qpos']],s['limits'],.2,polarity);sample.append(v)
                        for rows,value in zip(s['groups'],v):
                            if rows:current[rows,lane]+=float(value)
                    values.append(sample)
                state=cns.step_from_current(current.contiguous(),state,neutral=neutral)
                if not all(torch.isfinite(v).all().item() for v in state.fields()):raise RuntimeError('nonfinite CNS')
                rates=state.rates[neural_rows].cpu().numpy().copy();controls=(weights@rates).T
                for lane,(mode,_) in enumerate(conditions):
                    if mode=='zero-motor':controls[lane]=0
                history['motor_rates'].append(rates);history['controls'].append(controls.copy());history['sensory_current'].append(values)
                for b,control in zip(bodies,controls):
                    b.ctrl[:]=control;before=b.time
                    for _ in range(substeps):mujoco.mj_step(physics,b)
                    if not np.isfinite(b.qpos).all() or not np.isfinite(b.qvel).all() or abs(b.time-before-.01)>1e-10:
                        raise RuntimeError('nonfinite or reset physics')
                if tick%50==0:print(json.dumps(dict(tick=tick,up=history['thorax_up'][-1])),flush=True)
        report=dict(identity,completed=True,wall_seconds=time.monotonic()-began,simulated_seconds=history['time'][-1],
            final_up=history['thorax_up'][-1],unmapped_actuators=sum(not r['rows'] for r in recruitment),behavior_claim='none')
        (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('completed','wall_seconds','final_up','unmapped_actuators')},indent=2))
    except BaseException as error:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(error),samples=len(history['time']))));raise
    finally:
        np.savez_compressed(a.output/'trace.npz',**{k:np.asarray(v) for k,v in history.items()})


if __name__=='__main__':main()
