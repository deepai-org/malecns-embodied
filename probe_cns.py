"""Authenticate released V5 tensors and test CUDA recurrence, not behavior."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int, default=50)
args = parser.parse_args()
if args.steps <= 0 or args.output.exists():
    raise ValueError('positive steps and a fresh output path required')
sys.path.insert(0, str(args.source))
from chreatures.cns_adapter_contract import ARRAY_SPECS, validate_arrays
from research.anatomical_cns.model import AnatomicalCNS

manifest_bytes = (args.model / 'cns-manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
arrays = {}
for name, dtype, shape in ARRAY_SPECS:
    spec = manifest['buffers'][name]
    path = (args.model / spec['url']).resolve()
    if path.parent != args.model.resolve():
        raise ValueError('tensor path escapes model directory')
    transport = path.read_bytes()
    assert hashlib.sha256(transport).hexdigest() == spec['transportSha256'], name
    raw = gzip.decompress(transport)
    assert len(raw) == spec['byteLength'], name
    assert hashlib.sha256(raw).hexdigest() == spec['sha256'], name
    assert tuple(spec['shape']) == shape, name
    arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
validate_arrays(arrays)
print('Authenticated all V5 tensors', flush=True)
torch.set_num_threads(8)
device = torch.device('cuda:0')
model = AnatomicalCNS(arrays, device=device).eval()
# Two independent neural states: darkness and a uniform light field.
# Synthetic inputs are ONLY a diagnostic, not a body/environment simulation.
optic = torch.zeros((2, 1771, 3), device=device)
optic[1] = 0.5
body = torch.zeros((2, 807), device=device)
context = torch.zeros((2, 12), device=device)
assert np.count_nonzero(arrays['context.bias']) == 0
state = model.initial_state(2)
torch.cuda.synchronize()
started = time.monotonic()
with torch.inference_mode():
    for step in range(args.steps):
        latent, motor, state = model(optic, body, context, state)
        if not all(torch.isfinite(x).all().item() for x in (*state.fields(), motor, latent)):
            raise RuntimeError(f'nonfinite neural state at step {step}')
torch.cuda.synchronize()
elapsed = time.monotonic() - started
result = dict(
    kind='neural-only diagnostic; NOT embodied behavior',
    model_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    neurons=165122, edges=25563197, steps=args.steps, batch=2,
    simulated_seconds_per_state=args.steps * 0.01, wall_seconds=elapsed,
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    external_context='zero', all_finite=True,
    motor_min=motor.min().item(), motor_max=motor.max().item(),
    light_dark_motor_max_difference=(motor[0]-motor[1]).abs().max().item(),
    light_dark_rate_max_difference=(state.rates[:,0]-state.rates[:,1]).abs().max().item(),
)
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2), flush=True)
