"""Replay physical senses through fixed CNS with diagnostic body-current scaling.

Open-loop diagnostic, not an embodied controller or biological calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from model_loader import load_arrays


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'trial', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays, _, digest = load_arrays(a.model)
    receipt = json.loads((a.trial/'result.json').read_text())
    if receipt['model_manifest_sha256'] != digest or not receipt['completed']:
        raise ValueError('source trial/model mismatch')
    with np.load(a.trial/'trace.npz', allow_pickle=False) as trace:
        optic = trace['optic'][:-1, 0].copy()
        body = trace['body'][:-1, 0].copy()
        original = trace['motor_rates'][:, :, 0].copy()
    if original.shape != (len(optic), 815):
        raise ValueError('source motor trace must have time, neuron, resident axes')
    torch.set_num_threads(8)
    model = AnatomicalCNS(arrays, device=torch.device('cuda:0')).eval()
    motor_rows = torch.as_tensor(arrays['atlas.motor_rows'].astype(np.int64), device='cuda:0')
    body_rows = model.body_rows.long()
    gains = torch.tensor([0., 1., 10., 100.], device='cuda:0')
    state = model.initial_state(4)
    histories = {k: [] for k in ('motor_rates', 'body_current_rms', 'body_rate_rms', 'body_current_clipped_fraction')}
    started = time.monotonic()
    with torch.inference_mode():
        neutral = model.neutral_current(4).contiguous()
        baseline = state.rates.clone()
        for o, b in zip(optic, body):
            current = model.afferent_current(
                torch.as_tensor(o, device='cuda:0')[None].expand(4, -1, -1),
                torch.as_tensor(b, device='cuda:0')[None].expand(4, -1),
                torch.zeros((4, 12), device='cuda:0'))
            requested = neutral[body_rows] + (current[body_rows]-neutral[body_rows])*gains[None]
            histories['body_current_clipped_fraction'].append(((requested < 0) | (requested > 1)).float().mean(0).cpu().numpy())
            current[body_rows] = requested.clamp(0, 1)
            state = model.step_from_current(current.contiguous(), state, neutral=neutral)
            if not all(torch.isfinite(v).all().item() for v in state.fields()):
                raise RuntimeError('nonfinite CNS state')
            histories['motor_rates'].append(state.rates[motor_rows].T.cpu().numpy())
            histories['body_current_rms'].append(((current[body_rows]-neutral[body_rows]).square().mean(0).sqrt()).cpu().numpy())
            histories['body_rate_rms'].append(((state.rates[body_rows]-baseline[body_rows]).square().mean(0).sqrt()).cpu().numpy())
    histories = {k: np.asarray(v) for k, v in histories.items()}
    rates = histories['motor_rates']
    np.savez_compressed(a.output/'trace.npz', **histories, gains=gains.cpu().numpy(), motor_rows=motor_rows.cpu().numpy())
    summaries = []
    for lane, gain in enumerate(gains.cpu().tolist()):
        summaries.append(dict(body_current_gain=gain,
            maximum_motor_difference_from_body_neutral=float(np.abs(rates[:, lane]-rates[:, 0]).max()),
            late_maximum_individual_motor_temporal_std=float(rates[-100:, lane].std(0).max()),
            late_mean_body_current_rms=float(histories['body_current_rms'][-100:, lane].mean()),
            late_mean_body_rate_rms=float(histories['body_rate_rms'][-100:, lane].mean()),
            maximum_body_current_clipped_fraction=float(histories['body_current_clipped_fraction'][:, lane].max())))
    report = dict(kind='open-loop physical sensory transfer diagnostic', completed=True,
        model_manifest_sha256=digest, runner_sha256=sha(Path(__file__)),
        source_trace_sha256=sha(a.trial/'trace.npz'), source_receipt_sha256=sha(a.trial/'result.json'),
        simulated_seconds=len(optic)*.01, wall_seconds=time.monotonic()-started,
        resident=0, external_context='zero', neural_parameters='unchanged',
        original_replay_max_motor_error=float(np.abs(rates[:, 1]-original).max()),
        results=summaries,
        caveats=['Recorded physical senses, not closed-loop behavior.',
                 'Gains are deliberately uncalibrated diagnostic perturbations, not installed controller settings.',
                 'Body current alone is scaled around neutral and bounded to [0,1]; optic input is identical in every lane.',
                 'One recorded resident and one trajectory; not a general feasibility proof.'])
    with (a.output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
