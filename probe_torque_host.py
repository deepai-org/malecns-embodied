"""Verify the experimental host really removes servo forces, before CNS use."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from run_embodied import World, decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'scene', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    binary = a.source/'native/fly-world/target/release/chreatures-fly-world'
    fixture = json.loads(a.scene.read_text())
    records, reports = {}, {}
    for mode in ('zero-torque', 'pulse-torque', 'zero-position-command'):
        with (a.output/(mode+'.stderr.log')).open('w') as log:
            world = World(binary, a.scene, 23, log)
            try:
                ready = world.receive()
                if ready.get('experimental_torque_mode') != 'normalized-force-clamp-v1':
                    raise ValueError('wrong host capability')
                command = np.zeros((ready['residents'],92), dtype='<f4')
                qpos, force = [], []
                initial = world.rpc('sample')['sample']
                qpos.append(decode(initial,'qpos'))
                if mode == 'zero-torque':
                    invalid = command.copy(); invalid[0,84] = .1
                    try:
                        world.rpc('advance_torque', motor92_base64=base64.b64encode(invalid.tobytes()).decode())
                    except RuntimeError as error:
                        if 'requires channels 84..92 zero' not in str(error):
                            raise
                    else:
                        raise AssertionError('unsupported adhesion accepted')
                    assert world.rpc('sample')['sample']['time'] == initial['time']
                for tick in range(a.steps):
                    command[0,5] = .01 if mode == 'pulse-torque' and tick < 5 else 0
                    world.rpc('advance' if mode == 'zero-position-command' else 'advance_torque',
                              motor92_base64=base64.b64encode(command.tobytes()).decode())
                    sample = world.rpc('sample')['sample']
                    qpos.append(decode(sample,'qpos'))
                    force.append(decode(sample,'actuatorForce'))
                    if mode == 'zero-torque' or (mode == 'pulse-torque' and tick >= 5):
                        assert np.max(np.abs(force[-1])) == 0, 'residual actuator support'
                    if mode == 'pulse-torque' and tick < 5:
                        target = fixture['bodies'][0]['actuators'][5]
                        assert force[-1][target] > 0, 'requested torque absent'
                        others = np.delete(force[-1],target)
                        assert np.max(np.abs(others)) == 0, 'unrequested actuator support'
                records[mode+'_qpos'] = np.asarray(qpos)
                records[mode+'_force'] = np.asarray(force)
                reports[mode] = dict(maximum_actuator_force=float(np.max(np.abs(force))),
                                    final_time=sample['time'])
            finally:
                world.close()
    assert reports['zero-position-command']['maximum_actuator_force'] > 0
    np.savez_compressed(a.output/'trace.npz', **records)
    report = dict(completed=True, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
                  steps=a.steps, results=reports,
                  caveat='Host force-contract test only, no neural or behavioral claim')
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
