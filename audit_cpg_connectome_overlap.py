"""Register published CPG IDs/counts against the pinned full MaleCNS data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from scipy.sparse import coo_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','raw','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    expected={'annotations.feather':'2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2',
        'edges.feather':'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'}
    for n,h in expected.items():assert sha(a.raw/n)==h
    folder=a.source/'data/imac t1 connectome data'
    tablepath=folder/'wTable_20260210_vncRoisOnly.csv'
    wpath=folder/'W_20260210_vncRoisOnly.csv'
    with tablepath.open() as f:records=list(csv.DictReader(f))
    ids=np.array([int(r['bodyId']) for r in records])
    assert len(np.unique(ids))==len(ids) and np.all(np.diff(ids)>0)
    ann=feather.read_table(a.raw/'annotations.feather',columns=['bodyId','status']).to_pydict()
    traced=np.sort(np.array([i for i,s in zip(ann['bodyId'],ann['status']) if s=='Traced']))
    matched=np.isin(ids,traced)
    e=feather.read_table(a.raw/'edges.feather')
    pre=e['body_pre'].to_numpy();post=e['body_post'].to_numpy();count=e['weight'].to_numpy()
    keep=np.isin(pre,ids)&np.isin(post,ids)
    sub=coo_matrix((count[keep],(np.searchsorted(ids,pre[keep]),np.searchsorted(ids,post[keep]))),
                   shape=(len(ids),len(ids))).toarray()
    matrix=np.loadtxt(wpath,delimiter=',',skiprows=1)
    np.testing.assert_array_equal(matrix[:,0].astype(np.int64),ids)
    with wpath.open() as f:header=next(csv.reader(f))
    np.testing.assert_array_equal(np.array(header[1:],dtype=np.int64),ids)
    reference=np.abs(matrix[:,1:])
    union=(sub!=0)|(reference!=0)
    changed=(sub!=reference)&union
    thresholded=np.where(sub>=5,sub,0)
    threshold_union=(thresholded!=0)|(reference!=0)
    threshold_comparison=dict(threshold=5,
        official_nonzero_edges=int(np.count_nonzero(thresholded)),
        changed_pairs=int(np.count_nonzero((thresholded!=reference)&threshold_union)),
        only_official_edges=int(np.count_nonzero((thresholded!=0)&(reference==0))),
        only_reference_edges=int(np.count_nonzero((reference!=0)&(thresholded==0))),
        identical_nonzero_counts=int(np.count_nonzero((thresholded==reference)&threshold_union)),
        published_minimum_nonzero_count=float(reference[reference!=0].min()))
    examples=[]
    for i,j in np.argwhere(changed)[:20]:
        examples.append(dict(pre=int(ids[i]),post=int(ids[j]),
            published_absolute_weight=float(reference[i,j]),official_count=int(sub[i,j])))
    result=dict(kind='published subnetwork versus pinned whole-volume counts; not a replacement graph',
        raw_sha256=expected,reference_sha256={p.name:sha(p) for p in (tablepath,wpath)},
        neurons=len(ids),matched_traced_neurons=int(matched.sum()),
        missing_body_ids=ids[~matched].tolist(),
        reference_nonzero_edges=int(np.count_nonzero(reference)),
        official_induced_nonzero_edges=int(np.count_nonzero(sub)),
        identical_nonzero_counts=int(np.count_nonzero(union&~changed)),
        changed_pairs=int(changed.sum()),
        only_reference_edges=int(np.count_nonzero((reference!=0)&(sub==0))),
        only_official_edges=int(np.count_nonzero((reference==0)&(sub!=0))),
        examples=examples,threshold_comparison=threshold_comparison,runner_sha256=sha(Path(__file__)),
        caveats=['Absolute weights only: neurotransmitter sign agreement is not tested.',
            'Published filename says vncRoisOnly; official edges span the volume.',
            'Count differences do not by themselves imply erroneous data or identify their cause.',
            'Neuron ID overlap is not proof of equivalent connectivity or behavior.'])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','reference_sha256','raw_sha256')},indent=2))


if __name__=='__main__':main()
