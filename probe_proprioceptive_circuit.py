"""Test named afferent→motor pathways in the full graph, without body/decoder.

Predictions are anatomical circuit hypotheses from MANC, not an established
MaleCNS physiological ground truth. Injected current is in model units.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--sensory', type=Path, required=True)
    parser.add_argument('--motor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amplitude', type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.amplitude <= 1:
        raise ValueError('amplitude must be in (0,1] model units')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    from research.anatomical_cns.model import AnatomicalCNS
    arrays, _, model_hash = load_arrays(args.model)
    sensory = json.loads(args.sensory.read_text())
    motor = json.loads(args.motor.read_text())
    if sensory['source_sha256'] != motor['source_sha256']:
        raise ValueError('annotation sources differ')
    types = ['SNpp50', 'SNpp51', 'SNpp39', 'SNpp41']
    prediction = dict(SNpp50='extensor', SNpp51='flexor', SNpp39='flexor', SNpp41='extensor')
    groups = {}
    for kind in types:
        records = [r for r in sensory['neurons'] if r['type'] == kind and
                   r['entryNerve'] == 'ProLN' and (r['somaSide'] or r['rootSide']) == 'L']
        if not records:
            raise ValueError(f'no left-foreleg afferents for {kind}')
        groups[kind] = records
    targets = {}
    for kind, label in [('flexor', 'Ti flexor MN'), ('extensor', 'Ti extensor MN'),
                        ('accessory_flexor', 'Acc. ti flexor MN')]:
        targets[kind] = [r for r in motor['motors'] if r['mancType'] == label and
                        r['subclass'] == 'fl' and (r['somaSide'] or r['rootSide']) == 'L']
        if not targets[kind]:
            raise ValueError(f'no motor targets: {kind}')
    motor_rows = np.asarray(arrays['atlas.motor_rows'], dtype=np.int64)
    index = {int(row): i for i,row in enumerate(motor_rows)}
    for records in groups.values():
        for r in records:
            if r['cns_row'] not in arrays['atlas.body_rows'] or r['cns_row'] in index:
                raise ValueError('stimulation would violate afferent boundary')
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    model = AnatomicalCNS(arrays, device=device).eval()
    rows = torch.as_tensor(motor_rows, device=device)
    n = len(types)+1
    state = model.initial_state(n)
    with torch.inference_mode():
        neutral = model.neutral_current(n).contiguous()
        stimulation = torch.zeros_like(neutral)
        for condition, kind in enumerate(types, start=1):
            stimulation[[r['cns_row'] for r in groups[kind]], condition] = args.amplitude
        history = []
        start = time.monotonic()
        for tick in range(150):
            current = neutral + stimulation if 30 <= tick < 100 else neutral
            state = model.step_from_current(current.contiguous(), state, neutral=neutral)
            if not all(torch.isfinite(v).all().item() for v in state.fields()):
                raise RuntimeError(f'nonfinite state at {tick}')
            history.append(state.rates[rows].cpu().numpy())
    rates = np.asarray(history)
    # Subtract the matched unstimulated state at EACH timepoint. No assumed
    # spontaneous steady state or learned motor decoder enters these metrics.
    deltas = rates[:,:,1:] - rates[:,:,:1]
    result = {}
    for condition, kind in enumerate(types):
        changes = {}
        for target, records in targets.items():
            indices = [index[r['cns_row']] for r in records]
            changes[target] = float(deltas[80:100, indices, condition].mean())
        preference = changes['extensor']-changes['flexor']
        observed = 'extensor' if preference > 1e-9 else 'flexor' if preference < -1e-9 else 'unresolved'
        expected_signs = {
            'SNpp50': {'extensor': 1},
            'SNpp51': {'flexor': 1, 'extensor': -1, 'accessory_flexor': 1},
            'SNpp39': {'flexor': 1, 'extensor': -1, 'accessory_flexor': 1},
            'SNpp41': {'flexor': -1, 'extensor': 1, 'accessory_flexor': -1},
        }[kind]
        # SNpp50's delayed disinhibition does not justify a fixed flexor sign
        # in this arbitrary late window. Deliberately leave that sign untested.
        signed_match = {target: changes[target]*sign > 1e-9 for target,sign in expected_signs.items()}
        result[kind] = dict(stimulated_neurons=groups[kind],
            late_stimulation_motor_rate_change=changes,
            expected_preference=prediction[kind], observed_preference=observed,
            matches_relative_preference=observed == prediction[kind],
            expected_dominant_signs=expected_signs, matches_dominant_signs=signed_match,
            all_dominant_signs_match=all(signed_match.values()))
    np.savez_compressed(args.output/'trace.npz', motor_rates=rates, motor_rows=motor_rows,
                        time=np.arange(1,151)*.01)
    receipt = dict(kind='full-graph named proprioceptor perturbation; no behavioral claim',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_manifest_sha256=model_hash, amplitude_model_units=args.amplitude,
        neuron_count=165122, edge_count=25563197, source_annotation_sha256=sensory['source_sha256'],
        simulated_seconds=1.5, wall_seconds=time.monotonic()-start,
        stimulation_interval_seconds=[.3,1.0], comparison='unstimulated matched lane',
        readout_interval_seconds=[.8,1.0], external_context='zero', motor_targets=targets,
        source_hypothesis='https://elifesciences.org/reviewed-preprints/97766v1 Figure 59',
        caveats=['MANC cross-specimen circuit prediction, not measured MaleCNS reflex physiology.',
                 'Positive model-current pulses, not calibrated sensory transduction.',
                 'Different types contain different numbers of stimulated reconstructed cells.',
                 'Relative preference does not prove inhibition of the opposing motor group.',
                 'Dominant-sign summary does not test all predicted delayed or transient effects.',
                 'No body, muscle feedback or learned motor decoder participates.'], results=result)
    (args.output/'result.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='stimulated_neurons'} for k,v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
