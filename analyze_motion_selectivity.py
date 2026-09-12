"""Descriptive opposite-grating response asymmetry, separated by cell side."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','annotations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=json.loads((a.trial/'result.json').read_text())
    assert sha(a.annotations)==r['input_sha256']['annotations']
    table=feather.read_table(a.annotations)
    table=table.filter(pc.fill_null(pc.equal(table['status'],'Traced'),False)).sort_by([('bodyId','ascending')])
    cells=table.select(['bodyId','type','somaSide','rootSide','instance']).to_pylist()
    with np.load(a.trial/'trace.npz',allow_pickle=False) as t:
        rates=t['rates'].astype(np.float64);rows=t['rows'];times=t['time']
    assert r['conditions']==['gray','bright','dark','positive-phase','negative-phase']
    # Last full cycle of 2-Hz stimulation: no partial-cycle phase bias.
    assert np.allclose(times[80:130],np.arange(81,131)*.01)
    response=rates[80:130,:,3:5]-rates[80:130,:,:1]
    response-=response.mean(axis=0,keepdims=True)
    amplitude=np.sqrt(np.mean(response**2,axis=0))
    asymmetry=(amplitude[:,0]-amplitude[:,1])/np.maximum(amplitude.sum(axis=1),1e-15)
    results=[]
    for name in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'):
        for side in ('L','R',None):
            selected=[j for j,row in enumerate(rows) if cells[row]['type']==name and
                      (cells[row]['somaSide'] or cells[row]['rootSide'])==side]
            if not selected:continue
            valid=[j for j in selected if amplitude[j].sum()>1e-6]
            if not valid:raise ValueError('no above-floor responses')
            result=dict(type=name,side=side,count=len(selected),above_floor=len(valid),
                mean_positive_rms=float(amplitude[valid,0].mean()),mean_negative_rms=float(amplitude[valid,1].mean()),
                median_asymmetry=float(np.median(asymmetry[valid])),
                median_absolute_asymmetry=float(np.median(np.abs(asymmetry[valid]))),
                fraction_absolute_asymmetry_above_half=float(np.mean(np.abs(asymmetry[valid])>.5)))
            results.append(result)
    report=dict(kind=__doc__,runner_sha256=sha(Path(__file__)),
        trace_sha256=sha(a.trial/'trace.npz'),annotations_sha256=sha(a.annotations),results=results,
        method='Per-cell AC RMS over last complete cycle; (positive-negative)/(positive+negative), sum floor 1e-6.',
        caveats=['AC RMS is not a calcium-response direction-selectivity index or a validated physiological metric.',
                 'Camera-local phase directions have not been registered to each subtype preferred world direction.',
                 'Only two horizontal phase directions, one spatial/temporal frequency, one resting state.',
                 'Separate sides avoid cancellation, but a small difference here does not prove global absence of direction selectivity.',
                 'No model or stimulus parameter is changed by this analysis.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
