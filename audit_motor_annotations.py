"""Export exact annotated motor identities, without inferring muscle matches."""
import argparse
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
    raise ValueError('unexpected MaleCNS annotation source')
table = feather.read_table(args.annotations)
table = table.filter(pc.fill_null(pc.equal(table['status'], 'Traced'), False))
table = table.sort_by([('bodyId', 'ascending')])
fields = ['bodyId', 'type', 'instance', 'superclass', 'class', 'subclass',
          'somaSide', 'rootSide', 'entryNerve', 'exitNerve', 'mancType']
motors = []
for row, record in enumerate(table.select(fields).to_pylist()):
    if 'motor' in (record['superclass'] or ''):
        motors.append(dict(cns_row=row, **record))
result = dict(source_sha256=digest, traced_neurons=len(table),
              selection='Traced, sorted bodyId; superclass contains motor', motors=motors)
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(dict(traced_neurons=len(table), motor_neurons=len(motors),
    left_foreleg=[r for r in motors if r['subclass']=='fl' and (r['somaSide'] or r['rootSide'])=='L']), indent=2))
