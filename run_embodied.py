"""Measured closed-loop trial; no teacher, private context, or action selector.

The upstream motor decoder and effective joint servos remain explicit model
assumptions. A successful run does not itself establish useful fly behavior.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np
import torch
from model_loader import load_arrays


class World:
    def __init__(self, binary, scene, seed, log):
        self.process = subprocess.Popen(
            [str(binary), '--scene', str(scene), '--seed', str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        )
        self.buffer = b''
        self.sequence = 0

    def receive(self):
        deadline = time.monotonic() + 50
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('world response timeout; do not retry mutation')
            chunk = os.read(self.process.stdout.fileno(), 1048576)
            if not chunk:
                raise RuntimeError(f'world pipe closed: {self.process.poll()}')
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b'\n', 1)
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result)
        return result

    def rpc(self, command, **kwargs):
        self.sequence += 1
        self.process.stdin.write((json.dumps(dict(id=self.sequence, command=command, **kwargs))+'\n').encode())
        self.process.stdin.flush()
        result = self.receive()
        if result.get('id') != self.sequence:
            raise RuntimeError('world sequence mismatch')
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def decode(packet, name):
    spec = packet[name]
    result = np.frombuffer(base64.b64decode(spec['base64'], validate=True), dtype=spec['dtype']).copy()
    if result.size != spec['length'] or not np.isfinite(result).all():
        raise ValueError(f'invalid physical sample: {name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--condition', choices=['intact', 'sensory-disconnected', 'motor-disconnected'], default='intact')
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays, manifest, model_hash = load_arrays(args.model)
    scene = json.loads(args.scene.read_text())
    for model_key, scene_key in [('sensorySchema', 'cns_sensory_schema_sha256'),
                                  ('actuatorSchema', 'cns_actuator_schema_sha256'),
                                  ('morphology', 'body_schema_sha256')]:
        if manifest['identity'][model_key] != scene[scene_key]:
            raise ValueError(f'neural/physical schema mismatch: {model_key}')
    if np.count_nonzero(arrays['context.bias']):
        raise ValueError('strict trial requires zero external context bias')
    torch.set_num_threads(8)
    model = AnatomicalCNS(arrays, device=torch.device('cuda:0')).eval()
    binary = args.source / 'native/fly-world/target/release/chreatures-fly-world'
    identity = dict(condition=args.condition, model_manifest_sha256=model_hash,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                    steps=args.steps, seed=args.seed, external_context='zero',
                    caveat='Effective joint servos, not identified muscles; initialized weights')
    (args.output / 'intent.json').write_text(json.dumps(identity, indent=2))
    world = None
    records = {key: [] for key in ['time', 'optic', 'body', 'motor', 'delivered_motor', 'qpos', 'thorax_position', 'thorax_up', 'motor_rates']}
    began = time.monotonic()
    try:
        with (args.output / 'world.stderr.log').open('w') as log:
            world = World(binary, args.scene, args.seed, log)
            ready = world.receive()
            if ready.get('native_host') != 'chreatures-native-fly-world-v2':
                raise ValueError('unexpected physical protocol version')
            (args.output / 'world-ready.json').write_text(json.dumps(ready))
            count = ready['residents']
            context = torch.zeros((count, 12), device='cuda:0')
            state = model.initial_state(count)
            with torch.inference_mode():
                for tick in range(args.steps + 1):
                    packet = world.rpc('sample')['sample']
                    optic = decode(packet, 'optic').reshape(count, 1771, 3)
                    body = decode(packet, 'body').reshape(count, 807)
                    positions = decode(packet, 'bodyPositions').reshape(-1, 3)
                    rotations = decode(packet, 'bodyRotations').reshape(-1, 3, 3)
                    roots = [r['root_body_id'] for r in packet['bodyMap']['residents']]
                    records['time'].append(packet['time'])
                    records['optic'].append(optic)
                    records['body'].append(body)
                    records['qpos'].append(decode(packet, 'qpos'))
                    records['thorax_position'].append(positions[roots])
                    records['thorax_up'].append(rotations[roots, 2, 2])
                    if tick == args.steps:
                        break
                    neural_optic, neural_body = optic, body
                    if args.condition == 'sensory-disconnected':
                        neural_optic, neural_body = np.zeros_like(optic), np.zeros_like(body)
                    _, motor, state = model(torch.as_tensor(neural_optic, device='cuda:0'),
                        torch.as_tensor(neural_body, device='cuda:0'), context, state)
                    if not all(torch.isfinite(x).all().item() for x in (*state.fields(), motor)):
                        raise RuntimeError(f'nonfinite CNS at tick {tick}')
                    commands = motor.cpu().numpy().astype('<f4')
                    delivered = np.zeros_like(commands) if args.condition == 'motor-disconnected' else commands
                    records['motor'].append(commands.copy())
                    records['delivered_motor'].append(delivered.copy())
                    records['motor_rates'].append(state.rates[model.motor_rows.long()].cpu().numpy())
                    world.rpc('advance', motor92_base64=base64.b64encode(delivered.tobytes()).decode())
                    if tick % 25 == 0:
                        print(json.dumps(dict(tick=tick, time=packet['time'], thorax_up=rotations[roots,2,2].tolist())), flush=True)
            receipt = dict(identity, completed=True, wall_seconds=time.monotonic()-began,
                simulated_seconds=records['time'][-1]-records['time'][0],
                final_thorax_up=records['thorax_up'][-1].tolist(),
                displacement_mm=np.linalg.norm(records['thorax_position'][-1]-records['thorax_position'][0],axis=1).tolist(),
                behavior_claim='none; execution and trajectories only')
            (args.output / 'result.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt, indent=2), flush=True)
    except BaseException as error:
        (args.output / 'failure.json').write_text(json.dumps(dict(error=repr(error), samples=len(records['time']))))
        raise
    finally:
        if world:
            world.close()
        np.savez_compressed(args.output / 'trace.npz', **{k: np.asarray(v) for k,v in records.items()})


if __name__ == '__main__':
    main()
