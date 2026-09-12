"""Inventory exact MaleCNS sensory labels before defining transduction laws."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pyarrow.compute as pc
import pyarrow.feather as feather

parser = argparse.ArgumentParser()
parser.add_argument('annotations', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
digest = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
if digest != '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2':
    raise ValueError('unexpected annotation source')
table = feather.read_table(args.annotations)
table = table.filter(pc.fill_null(pc.equal(table['status'], 'Traced'), False))
table = table.sort_by([('bodyId', 'ascending')])
fields = ['bodyId', 'type', 'instance', 'superclass', 'class', 'subclass',
          'somaSide', 'rootSide', 'entryNerve', 'mancType']
selected = []
for row, item in enumerate(table.select(fields).to_pylist()):
    if item['subclass'] == 'chordotonal organ':
        selected.append(dict(cns_row=row, **item))
result = dict(source_sha256=digest, neurons=selected,
    caveat='Inventory only; no functional tuning inferred from row order or IDs')
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(dict(count=len(selected),
    types=dict(Counter(r['type'] for r in selected)),
    nerves=dict(Counter(str(r['entryNerve']) for r in selected))), indent=2))
