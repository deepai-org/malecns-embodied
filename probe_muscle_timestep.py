"""Isolate muscle-force update timing in a static whole-body physical scene.

No CNS or aerodynamics: replay recorded activations to compare held torque
against force evaluated each physics step. This is a numerical diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    import mujoco
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=.2)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    receipt = json.loads((a.trial/'intent.json').read_text())
    scene = json.loads(a.scene.read_text())
    assert receipt['scene_sha256']==hashlib.sha256(a.scene.read_bytes()).hexdigest()
    with np.load(a.trial/'trace.npz',allow_pickle=False) as trace:
        activations = trace['activation'].copy()
        initial_qpos,initial_qvel = trace['qpos'][0].copy(),trace['qvel'][0].copy()
    results,histories = {},{}
    for mode in ('held-10ms','updated-0.1ms','zero'):
        model = mujoco.MjModel.from_xml_path(str(a.scene.parent/scene['scene_xml']))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0)
        data.qpos[:],data.qvel[:] = initial_qpos,initial_qvel
        limits = model.actuator_forcerange.copy()
        model.actuator_forcerange[:] = 0
        data.ctrl[:] = 0
        interval = round(.01/model.opt.timestep)
        if not np.isclose(interval*model.opt.timestep,.01,rtol=0,atol=1e-12):
            raise ValueError('unexpected physics timestep')
        records,failed = [],False
        for step in range(round(a.seconds/model.opt.timestep)):
            if mode!='zero' and (mode=='updated-0.1ms' or step%interval==0):
                activation = activations[min(step//interval,len(activations)-1)]
                for lane,resident in enumerate(scene['bodies']):
                    for muscle,g in enumerate(receipt['muscle_groups']):
                        flexor,extensor = activation[lane,muscle]
                        pos,vel = g['addresses'][lane]
                        normalized = np.clip(.03*receipt['polarity']*(flexor-extensor)
                                             + .03*(flexor+extensor)*(g['rest']-data.qpos[pos])
                                             - .001*(flexor+extensor)*data.qvel[vel],-1,1)
                        actuator = resident['actuators'][g['channel']]
                        capacity = limits[actuator,1] if normalized>=0 else -limits[actuator,0]
                        model.actuator_forcerange[actuator] = normalized*capacity
            before = data.time
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isclose(data.time-before,model.opt.timestep,rtol=0,atol=1e-12):
                failed = True
                break
            records.append([data.time,float(np.max(np.abs(data.qvel))),float(np.max(np.abs(data.actuator_force)))])
        histories[mode] = np.asarray(records)
        results[mode] = dict(failed_or_reset=failed,completed_steps=len(records),
                            completed_seconds=len(records)*model.opt.timestep,
                            maximum_recorded_speed=float(np.max(histories[mode][:,1])) if records else None)
    np.savez_compressed(a.output/'trace.npz',**histories)
    report = dict(kind='physical force-update timing diagnostic, not CNS behavior',results=results,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scene_sha256=receipt['scene_sha256'],
                  source_trace_sha256=hashlib.sha256((a.trial/'trace.npz').read_bytes()).hexdigest(),
                  caveats=['Static scene: no native aerodynamics or ecology updates.',
                           'Recorded activation prefix is replayed, then last activation held.',
                           'Force uses original capacity, not native metabolic capacity scaling.',
                           'Updated forces are still explicit; this does not prove numerical convergence.'])
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
