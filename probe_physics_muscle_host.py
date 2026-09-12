"""Bounded protocol/force checks for the new physics-step muscle interface."""
import argparse
import base64
import json
import hashlib
from pathlib import Path
import numpy as np
from run_embodied import World,decode


def encoded(array):
    return base64.b64encode(np.asarray(array,dtype='<f4').tobytes()).decode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','scene','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    binary=a.source/'native/fly-world/target/release/chreatures-fly-world'
    samples={}
    for mode in ('zero','constant','held-constant'):
        with (a.output/(mode+'.stderr.log')).open('w') as log:
            world=World(binary,a.scene,23,log)
            try:
                ready=world.receive()
                assert ready.get('experimental_muscle_mode')=='affine-force-physics-step-v1'
                count=ready['residents'];coefficients=np.zeros((count,84,3),dtype='<f4')
                if mode=='zero':
                    initial=world.rpc('sample')['sample']['time']
                    for value in (-.1,float('nan')):
                        invalid=coefficients.copy();invalid[0,0,1]=value
                        try:
                            world.rpc('advance_muscle',muscle252_base64=encoded(invalid))
                        except RuntimeError as error:
                            assert 'coefficient' in str(error),str(error)
                        else:
                            raise AssertionError('invalid coefficient accepted')
                        assert world.rpc('sample')['sample']['time']==initial
                if mode=='held-constant':
                    command=np.zeros((count,92),dtype='<f4');command[0,5]=.01
                    world.rpc('advance_torque',motor92_base64=encoded(command))
                else:
                    if mode=='constant':coefficients[0,5,0]=.01
                    world.rpc('advance_muscle',muscle252_base64=encoded(coefficients))
                packet=world.rpc('sample')['sample']
                samples[mode+'_qpos']=decode(packet,'qpos')
                samples[mode+'_force']=decode(packet,'actuatorForce')
                assert np.isclose(packet['time'],.01,rtol=0,atol=1e-12)
            finally:
                world.close()
    np.testing.assert_array_equal(samples['zero_force'],0)
    np.testing.assert_array_equal(samples['constant_qpos'],samples['held-constant_qpos'])
    np.testing.assert_array_equal(samples['constant_force'],samples['held-constant_force'])
    np.savez_compressed(a.output/'trace.npz',**samples)
    result=dict(completed=True,zero_force_exact=True,constant_force_first_tick_equivalent=True,
                invalid_coefficients_rejected_without_time_advance=True,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                caveat='One-tick constant-force equivalence does not test dynamic muscle convergence or metabolism.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
