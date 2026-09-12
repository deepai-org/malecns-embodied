"""Load authenticated upstream browser tensors into the upstream Torch model."""
import gzip
import hashlib
import json
import numpy as np


def load_arrays(directory):
    from chreatures.cns_adapter_contract import ARRAY_SPECS, validate_arrays
    manifest_bytes = (directory / 'cns-manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    arrays = {}
    for name, dtype, shape in ARRAY_SPECS:
        spec = manifest['buffers'][name]
        path = (directory / spec['url']).resolve()
        if path.parent != directory.resolve():
            raise ValueError('tensor path escapes model directory')
        transport = path.read_bytes()
        if hashlib.sha256(transport).hexdigest() != spec['transportSha256']:
            raise ValueError(f'transport checksum: {name}')
        raw = gzip.decompress(transport)
        if len(raw) != spec['byteLength'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError(f'tensor checksum: {name}')
        if tuple(spec['shape']) != shape:
            raise ValueError(f'tensor shape: {name}')
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    validate_arrays(arrays)
    return arrays, manifest, hashlib.sha256(manifest_bytes).hexdigest()
