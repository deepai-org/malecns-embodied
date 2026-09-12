"""Whole-body exploratory antagonist actuation; unresolved joints unpowered.

This replaces the effective position servos, not the remaining approximate
sensory adapter or neural physiology. No rich-behavior claim follows from a run.
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
from aggregate_leg_muscles import AggregateLegMuscles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','scene','motor','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--polarity',type=int,choices=(-1,1),default=1)
    p.add_argument('--condition',choices=('intact','frozen-sensory','zero-motor'),default='intact')
    a = p.parse_args()
    if a.steps <= 0:
        raise ValueError('steps must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays,manifest,model_hash = load_arrays(a.model)
    scene = json.loads(a.scene.read_text())
    for model_key, scene_key in [('sensorySchema','cns_sensory_schema_sha256'),
                                 ('actuatorSchema','cns_actuator_schema_sha256'),
                                 ('morphology','body_schema_sha256')]:
        if manifest['identity'][model_key] != scene[scene_key]:
            raise ValueError('neural/physical schema mismatch')
    if np.count_nonzero(arrays['context.bias']):
        raise ValueError('nonzero context bias')
    annotation = json.loads(a.motor.read_text())
    drive = AggregateLegMuscles(annotation,arrays['atlas.motor_rows'],scene,a.polarity)
    binary = a.source/'native/fly-world/target/release/chreatures-fly-world'
    identity = dict(condition=a.condition,polarity=a.polarity,steps=a.steps,
                    model_manifest_sha256=model_hash,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    muscle_code_sha256=hashlib.sha256(Path(__file__).with_name('aggregate_leg_muscles.py').read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
                    motor_annotation_sha256=hashlib.sha256(a.motor.read_bytes()).hexdigest(),
                    muscle_groups=drive.groups,
                    caveats=['24 aggregate muscle units across 12 hinges; 72 other joint actuators unpowered.',
                             'Fixed rig-rest angle and contractile/stiffness/damping gains are uncalibrated assumptions.',
                             'Flexor/extensor coordinate sign is a hypothesis, not validated muscle geometry.',
                             'Native adhesion and oral channels zero. Native passive mechanics and aerodynamics remain.',
                             'Native ecological work/capacity model was designed for servos; not a validated torque metabolism.',
                             'Original broad sensory adapter remains; no learned motor decoder contributes to actuation.',
                             'Four residents share one scene, not independent replicates.'])
    (a.output/'intent.json').write_text(json.dumps(identity,indent=2))
    torch.set_num_threads(8)
    cns = AnatomicalCNS(arrays,device=torch.device('cuda:0')).eval()
    history = {k:[] for k in ('time','qpos','qvel','optic','body','motor_rates','command','activation','actuator_force','thorax_position','thorax_up')}
    world = None
    began = time.monotonic()
    try:
        with (a.output/'world.stderr.log').open('w') as log:
            world = World(binary,a.scene,23,log)
            ready = world.receive()
            if ready.get('experimental_torque_mode') != 'normalized-force-clamp-v1':
                raise ValueError('wrong torque host')
            count = ready['residents']
            if count != len(scene['residents']):
                raise ValueError('resident count changed')
            state = cns.initial_state(count)
            context = torch.zeros((count,12),device='cuda:0')
            with torch.inference_mode():
                for tick in range(a.steps+1):
                    sample = world.rpc('sample')['sample']
                    optic = decode(sample,'optic').reshape(count,1771,3)
                    body = decode(sample,'body').reshape(count,807)
                    qpos,qvel = decode(sample,'qpos'),decode(sample,'qvel')
                    roots = [r['root_body_id'] for r in sample['bodyMap']['residents']]
                    history['time'].append(sample['time'])
                    history['qpos'].append(qpos); history['qvel'].append(qvel)
                    history['optic'].append(optic); history['body'].append(body)
                    history['actuator_force'].append(decode(sample,'actuatorForce'))
                    history['thorax_position'].append(decode(sample,'bodyPositions').reshape(-1,3)[roots])
                    history['thorax_up'].append(decode(sample,'bodyRotations').reshape(-1,3,3)[roots,2,2])
                    if tick == 0:
                        frozen_optic,frozen_body = optic.copy(),body.copy()
                    if tick == a.steps:
                        break
                    inputs = (frozen_optic,frozen_body) if a.condition=='frozen-sensory' else (optic,body)
                    _,_,state = cns(torch.as_tensor(inputs[0],device='cuda:0'),
                                    torch.as_tensor(inputs[1],device='cuda:0'),context,state)
                    if not all(torch.isfinite(x).all().item() for x in state.fields()):
                        raise RuntimeError('nonfinite CNS')
                    rates = state.rates[cns.motor_rows.long()].cpu().numpy().copy()
                    command = drive.step(rates,qpos,qvel,zero_motor=a.condition=='zero-motor')
                    history['motor_rates'].append(rates)
                    history['command'].append(command.copy())
                    history['activation'].append(drive.activation.copy())
                    world.rpc('advance_torque',motor92_base64=base64.b64encode(command.astype('<f4').tobytes()).decode())
                    if tick%50 == 0:
                        print(json.dumps(dict(tick=tick,up=history['thorax_up'][-1].tolist())),flush=True)
        result = dict(identity,completed=True,wall_seconds=time.monotonic()-began,
                      simulated_seconds=history['time'][-1]-history['time'][0],
                      final_thorax_up=history['thorax_up'][-1].tolist(),
                      displacement_mm=np.linalg.norm(history['thorax_position'][-1]-history['thorax_position'][0],axis=1).tolist(),
                      maximum_command=float(np.max(np.abs(history['command']))),behavior_claim='none')
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ('completed','final_thorax_up','displacement_mm','maximum_command')},indent=2),flush=True)
    except BaseException as error:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(error),samples=len(history['time']))))
        raise
    finally:
        if world:
            world.close()
        np.savez_compressed(a.output/'trace.npz',**{k:np.asarray(v) for k,v in history.items()})


if __name__ == '__main__':
    main()
