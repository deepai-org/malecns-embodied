"""Mean-parameter reference probe of Pugliese et al.'s supplied mCNS network.

Independent SciPy integration, not reproduction of their randomized ensemble.
No body, sensory adapter, neural training, or external rhythmic input.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix

PIN = 'faee4b06869855ae0164cbf217fb6ec28ef3521b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rtol',type=float,default=2e-6)
    p.add_argument('--atol',type=float,default=5e-9)
    a = p.parse_args()
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip() == PIN
    a.output.mkdir(parents=True,exist_ok=False)
    folder = a.source/'data/imac t1 connectome data'
    tablepath = folder/'wTable_20260210_vncRoisOnly.csv'
    wpath = folder/'W_20260210_vncRoisOnly.csv'
    with tablepath.open() as f:
        table = list(csv.DictReader(f))
    ids = np.array([int(r['bodyId']) for r in table])
    with wpath.open() as f:
        header = next(csv.reader(f))
    np.testing.assert_array_equal(np.array(header[1:],dtype=np.int64),ids)
    raw = np.loadtxt(wpath,delimiter=',',skiprows=1)
    np.testing.assert_array_equal(raw[:,0].astype(np.int64),ids)
    # Upstream reweight_connectivity transposes pre-by-post W and uses .03 for both signs.
    w = csr_matrix(raw[:,1:].T*.03)
    del raw
    dn = [i for i,r in enumerate(table) if r['type']=='DNg100']
    motor = [i for i,r in enumerate(table) if r['class']=='motor neuron' or r['superclass']=='vnc_motor']
    assert len(dn)==2 and motor
    coretypes = {'IN17A001','INXXX466','IN19B012','IN16B036','IN19A007'}
    selected = sorted(set(motor+dn+[i for i,r in enumerate(table) if r['type'] in coretypes]))
    timepoints = np.linspace(.02,1.999,1980)
    traces, summaries = [], []
    start = time.monotonic()
    for target in [None]+dn:
        current = np.zeros(len(ids))
        if target is not None:
            current[target] = 250.
        def rhs(t,r):
            activation = np.maximum(200*np.tanh((current+w@r-7.5)/200),0)
            return (activation-r)/.02
        sol = solve_ivp(rhs,(.02,1.999),np.zeros(len(ids)),t_eval=timepoints,
                        rtol=a.rtol,atol=a.atol,method='RK45',max_step=.001)
        assert sol.success and sol.y.shape == (len(ids),len(timepoints))
        assert np.isfinite(sol.y).all() and sol.y.min() >= -1e-5, (float(sol.y.min()),float(sol.y.max()))
        traces.append(sol.y[selected].astype(np.float32))
        late = timepoints >= .5
        metrics=[]
        for row in motor:
            values = sol.y[row,late]
            peaks,_ = find_peaks(values,prominence=1.)
            metrics.append(dict(body_id=int(ids[row]),type=table[row]['type'],
                peak_to_peak=float(np.ptp(values)),peak_count=int(len(peaks)),
                mean_rate=float(np.mean(values)),
                mean_peak_frequency_hz=float(1/np.mean(np.diff(timepoints[late][peaks]))) if len(peaks)>1 else None))
        summaries.append(dict(stimulated_body_id=int(ids[target]) if target is not None else None,
            function_evaluations=sol.nfev,min_rate=float(sol.y.min()),max_rate=float(sol.y.max()),motor_metrics=metrics,
            motors_with_at_least_3_peaks=sum(m['peak_count']>=3 for m in metrics)))
        print(json.dumps({k:v for k,v in summaries[-1].items() if k!='motor_metrics'}),flush=True)
    np.savez_compressed(a.output/'trace.npz',rates=np.stack(traces),time=timepoints,
                        body_ids=ids[selected],motor_body_ids=ids[motor])
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(completed=True,source_repository='https://github.com/smpuglie/Pugliese_2026',
        source_revision=PIN,runner_sha256=sha(Path(__file__)),
        input_sha256={str(path.relative_to(a.source)):sha(path) for path in
            (tablepath,wpath,a.source/'src/simulation/vnc_sim.py',a.source/'configs/neuron_params/default.yaml')},
        neurons=len(ids),edges=w.nnz,motors=len(motor),rtol=a.rtol,atol=a.atol,max_step=.001,
        parameters=dict(tau=.02,gain=1,threshold=7.5,rate_cap=200,synapse_multiplier=.03,current=250),
        wall_seconds=time.monotonic()-start,conditions=summaries,trace_sha256=sha(a.output/'trace.npz'),
        caveats=['Published mCNS front-leg subnetwork, not our full 165122-cell artifact.',
            'Mean parameters, no cell-size scaling or random draws; not the published ensemble.',
            'Constant DN input over recorded interval; zero initial state at pulse onset .02s.',
            'Peak criterion is descriptive: >=3 late peaks with 1 model-rate-unit prominence.',
            'No external oscillator, body, muscle commands, or rich-behavior claim.'])
    (a.output/'result.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
