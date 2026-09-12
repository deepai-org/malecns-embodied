"""Bounded offline pose-and-muscle equilibrium search; never a controller."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from pooled_muscle_mapping import POOLS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','poses','motor','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=120)
    a=p.parse_args();poses=json.loads(a.poses.read_text())
    assert sha(a.xml)==poses['xml_sha256']
    m=mujoco.MjModel.from_xml_path(str(a.xml));d=mujoco.MjData(m)
    assert m.nq==49 and m.nv==48 and m.nu==m.na==90
    motor=json.loads(a.motor.read_text())['motors']
    weights=np.zeros((m.nu,len(motor)))
    for i in range(m.nu):
        tag,name=m.actuator(i).name.split('/',1)
        selected=[j for j,r in enumerate(motor) if r['mancType'] in POOLS[name]
            and r['subclass']=={'f':'fl','m':'ml','h':'hl'}[tag[1]]
            and (r['somaSide'] or r['rootSide'])==tag[0].upper()]
        if selected:weights[i,selected]=1/len(selected)
    # Neurons with identical recruitment columns are interchangeable here.
    # Replacing their individual inputs by their mean exactly preserves the
    # achievable activation set for independent inputs in [0,1]. No new wiring.
    groups={}
    for j in range(len(motor)):
        if np.any(weights[:,j]):groups.setdefault(tuple(weights[:,j]),[]).append(j)
    grouped=np.stack([np.asarray(column)*len(indices) for column,indices in groups.items()],axis=1)
    ngroup=grouped.shape[1]
    feet=[m.geom(tag+'/LFTarsus5_geom').id for tag in ('lf','lm','lh','rf','rm','rh')]
    vertices=[]
    for g in feet:
        mesh=m.geom_dataid[g];start,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
        vertices.append(m.mesh_vert[start:start+count].copy())
    pairs=[(i,j) for i in range(m.ngeom) for j in range(i+1,m.ngeom)]
    pair_index={pair:i for i,pair in enumerate(pairs)}
    weight=float(m.body_mass.sum()*np.linalg.norm(m.opt.gravity))
    scale=np.r_[np.full(3,weight),np.full(3,weight*.5),np.full(42,weight*.05)]
    joint_ids=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
    assert [int(m.jnt_qposadr[j]) for j in joint_ids]==list(range(7,49))
    lower=np.r_[.9,m.jnt_range[joint_ids,0],np.zeros(ngroup+24)]
    upper=np.r_[2.3,m.jnt_range[joint_ids,1],np.ones(ngroup),np.full(24,2.)]
    began=time.monotonic();results=[]
    for pose_index,pose in enumerate(poses['results']):
        base=np.asarray(pose['qpos'])
        def evaluate(x):
            d.qpos[:]=base;d.qpos[2]=x[0];d.qpos[7:]=x[1:43]
            d.qvel[:]=0;d.ctrl[:]=0;d.act[:]=grouped@x[43:43+ngroup]
            mujoco.mj_forward(m,d)
            balance=d.qfrc_actuator+d.qfrc_passive-d.qfrc_bias
            penetration=np.zeros(len(pairs))
            for c in d.contact:
                k=pair_index[tuple(sorted(map(int,c.geom)))]
                penetration[k]=max(penetration[k],-float(c.dist))
            heights=[]
            for i,(g,v) in enumerate(zip(feet,vertices)):
                world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                point=world[np.argmin(world[:,2])];heights.append(point[2])
                jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[g]))
                mu=float(max(m.geom_friction[g,0],m.geom_friction[m.geom('floor').id,0]))
                directions=np.array([[mu,0,1],[-mu,0,1],[0,mu,1],[0,-mu,1.]])
                balance+=jp.T@(weight*x[43+ngroup+4*i:43+ngroup+4*(i+1)]@directions)
            return balance,penetration,np.asarray(heights)
        def objective(x):
            balance,penetration,heights=evaluate(x)
            return np.r_[balance/scale,100*penetration,100*heights]
        start=np.r_[base[2],base[7:],np.full(ngroup,.2),np.full(24,1/24)]
        start=np.clip(start,lower+1e-9,upper-1e-9)
        fit=least_squares(objective,start,bounds=(lower,upper),max_nfev=a.max_nfev,
                          ftol=1e-8,xtol=1e-8,gtol=1e-8,tr_solver='lsmr')
        balance,penetration,heights=evaluate(fit.x)
        geometric=bool(penetration.max()<1e-5 and abs(heights).max()<1e-4)
        supported=bool(geometric and abs(balance).max()<1e-5)
        result=dict(pose_index=pose_index,status=int(fit.status),message=fit.message,
            nfev=int(fit.nfev),cost=float(fit.cost),qpos=d.qpos.tolist(),
            geometric_candidate=geometric,supported_candidate=supported,
            maximum_balance_error=float(abs(balance).max()),balance_residual=balance.tolist(),
            maximum_penetration=float(penetration.max()),foot_bottom_z=heights.tolist(),
            group_inputs=fit.x[43:43+ngroup].tolist(),activations=d.act.tolist(),
            support_ray_weights=(weight*fit.x[43+ngroup:]).tolist())
        results.append(result)
        print(json.dumps({k:result[k] for k in ('pose_index','nfev','maximum_balance_error','maximum_penetration','geometric_candidate','supported_candidate')}),flush=True)
    report=dict(kind=__doc__,runner_sha256=sha(Path(__file__)),xml_sha256=sha(a.xml),
        inputs={name:sha(getattr(a,name)) for name in ('poses','motor')},
        pooling_sha256=sha(Path(__file__).with_name('pooled_muscle_mapping.py')),
        mujoco=mujoco.__version__,wall_seconds=time.monotonic()-began,max_nfev=a.max_nfev,
        grouped_recruitment=grouped.tolist(),results=results,
        assumptions=['Same body, collision masks, passive forces, named recruitment, and muscle parameters.',
            'Search varies initial root height and joint pose, bounded muscle inputs, and ideal foot forces.',
            'Uses declared joint ranges even where source runtime limits are disabled.',
            'One ideal point contact per foot with four friction rays; no torso/self-contact force in equilibrium.',
            'Foot-force rays bounded to twice total weight each as an additional search restriction.',
            'Candidate thresholds: balance <1e-5, penetration <1e-5, foot height error <1e-4 model units.',
            'Local-search failure is not global infeasibility. No trajectory or CNS activity is optimized.',
            'No fitted input or force is supplied to a running fly or installed as a controller.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
