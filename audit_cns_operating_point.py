"""Audit the authenticated initialized CNS, without tuning it for behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from model_loader import load_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.source))
    arrays,_,digest=load_arrays(a.model)
    effective={}
    for name,offset,span in (('baseline_raw',.05,.4),('recurrent_gain_raw',.5,1.5),
                             ('tau_raw',.02,.23),('adaptation_gain_raw',0,.5),
                             ('adaptation_tau_raw',.25,4.75)):
        raw=arrays['dynamics.'+name].astype(np.float64)
        values=offset+span/(1+np.exp(-raw))
        effective[name]=dict(parameter_count=raw.size,minimum=float(values.min()),
                             maximum=float(values.max()),distinct_values=int(len(np.unique(values))))
    weight=arrays['graph.weight_bits'].view('<f2').astype(np.float64)
    channels=arrays['graph.channel'][arrays['graph.col']]
    prefix=np.empty(len(weight)+1,dtype=np.float64);prefix[0]=0
    np.cumsum(np.abs(weight)*(channels==1),out=prefix[1:])
    ptr=arrays['graph.crow'].astype(np.int64)
    row_mass=prefix[ptr[1:]]-prefix[ptr[:-1]]
    raw=arrays['dynamics.recurrent_gain_raw'].astype(np.float64)
    gain=(.5+1.5/(1+np.exp(-raw)))[arrays['atlas.neuron_type']]
    report=dict(kind='initialized CNS operating-point audit; no behavior claim',
        model_manifest_sha256=digest,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_model_code_sha256=hashlib.sha256((a.source/'research/anatomical_cns/model.py').read_bytes()).hexdigest(),
        effective_parameters=effective,
        fast_recurrent_maximum_absolute_row_sum=float(row_mass.max()),
        fast_recurrent_gain_weighted_infinity_norm_bound=float((row_mass*gain).max()),
        source_channel_edge_counts={str(int(c)):int((channels==c).sum()) for c in np.unique(channels)},
        body_weight_standard_deviation=float(arrays['body.weight'].std()),
        interpretation=[
            'At exact neutral input and initialized state, all recurrent deviations, adaptation and modulation are zero; the deterministic update remains at baseline.',
            'The row-sum bound concerns only the fast recurrent operator at initialization. It is not a stability proof for the full adaptive, modulated, plastic or embodied system.',
            'Identical effective values across parameter slots are initialization choices, not cell-specific physiological calibration.',
            'This audit does not prove the model cannot behave under suitable inputs or different biologically justified parameters.'])
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
