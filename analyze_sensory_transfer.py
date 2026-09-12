"""Check sensory replay provenance and quantify named antagonist separation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--source-trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = json.loads((a.trial/'result.json').read_text())
    source = json.loads((a.source_trial/'result.json').read_text())
    assert r['completed'] and r['model_manifest_sha256'] == source['model_manifest_sha256']
    assert r['runner_sha256'] == sha(Path(__file__).with_name('probe_sensory_transfer.py'))
    assert r['source_trace_sha256'] == sha(a.source_trial/'trace.npz')
    assert r['source_receipt_sha256'] == sha(a.source_trial/'result.json')
    with np.load(a.trial/'trace.npz', allow_pickle=False) as t:
        assert all(np.isfinite(t[k]).all() for k in t.files)
        rates, gains = t['motor_rates'].copy(), t['gains'].copy()
    with np.load(a.source_trial/'trace.npz', allow_pickle=False) as t:
        original = t['motor_rates'][:, :, r['resident']].copy()
    assert rates.shape == (len(original), 4, 815)
    error = float(np.abs(rates[:, 1]-original).max())
    assert error == r['original_replay_max_motor_error'] and error < 1e-6
    groups = source['muscle_groups']
    separation = np.stack([rates[:, :, g['cohorts'][0]].mean(2)-rates[:, :, g['cohorts'][1]].mean(2) for g in groups], axis=2)
    report = dict(checked=True, replay_max_motor_error=error,
        analyzer_sha256=sha(Path(__file__)), trace_sha256=sha(a.trial/'trace.npz'),
        late_window_seconds=[3., 4.],
        results=[dict(body_current_gain=float(gain),
                      maximum_antagonist_separation=float(np.abs(separation[:, lane]).max()),
                      late_maximum_antagonist_separation=float(np.abs(separation[-100:, lane]).max()),
                      late_maximum_antagonist_temporal_std=float(separation[-100:, lane].std(0).max()))
                 for lane, gain in enumerate(gains)],
        caveat='Named mean motor-rate differences before muscle recruitment; no behavior or calibrated strength claim.')
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
