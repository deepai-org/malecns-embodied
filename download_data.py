"""Download and authenticate official public MaleCNS v1.0 inputs (no credentials)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations.feather': ('body-annotations-male-cns-v1.0-minconf-0.5.feather', 14483314, '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'),
    'neurotransmitters.feather': ('body-neurotransmitters-male-cns-v1.0.feather', 43282834, '95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621'),
    'edges.feather': ('connectome-weights-male-cns-v1.0-minconf-0.5.feather', 1051241946, 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, (source, size, expected) = item
        path = args.output / name
        candidate = path if path.exists() else path.with_suffix('.partial')
        if candidate != path:
            with urllib.request.urlopen(BASE+source, timeout=90) as response, candidate.open('xb') as target:
                while chunk := response.read(8*1024*1024):
                    target.write(chunk)
        with candidate.open('rb') as stream:
            observed = hashlib.file_digest(stream, 'sha256').hexdigest()
        if candidate.stat().st_size != size or observed != expected:
            raise ValueError(f'Checksum mismatch: {candidate}; retained for inspection')
        if candidate != path:
            candidate.rename(path)
        print('VERIFIED', name, size, flush=True)
        return name, dict(url=BASE+source, bytes=size, sha256=expected)

    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = dict(pool.map(fetch, FILES.items()))
    receipt = dict(dataset='MaleCNS v1.0', source='https://male-cns.janelia.org/download/',
                   license='CC-BY-4.0', files=verified)
    (args.output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
