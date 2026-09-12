"""Index published physiological records; does not download or invent measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

PIN = 'c68159f1f7a4ecc957c708ae8411fd2550482f63'
RECORD = 'Records/Azevedo_2020_Records/Dataset2_SlowInterFast_SensoryFeedback.m'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    revision = subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    assert revision == PIN
    raw = (a.source/RECORD).read_bytes()
    cells = []
    for line in raw.decode().splitlines():
        match = re.match(r"^T\{(?:1|2|end\+1),:\}\s*=\s*\{'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)'",line)
        if not match:
            continue
        cell,genotype,kind,protocol = match.groups()
        # Mirrors Script_showSensoryFeedbackPeaksVsSpeedAndPosition.m line 51.
        included = 'iav-LexA' not in genotype and not ('ChR' in genotype and '81A07' in genotype)
        cells.append(dict(cell_id=cell,genotype=genotype,motor_class=kind,
            protocol=protocol,in_main_membrane_potential_comparison=included))
    assert cells and len({c['cell_id'] for c in cells}) == len(cells)
    files, pages = {}, []
    for page in (1,2,3):
        url = f'https://datadryad.org/api/v2/versions/105831/files?page={page}'
        with urllib.request.urlopen(url,timeout=30) as response:
            data = response.read()
        pages.append(dict(url=url,sha256=hashlib.sha256(data).hexdigest()))
        for f in json.loads(data)['_embedded']['stash:files']:
            files[f['path']] = dict(bytes=f['size'],
                metadata_url='https://datadryad.org'+f['_links']['self']['href'],
                api_download_url='https://datadryad.org'+f['_links']['stash:download']['href'])
    for cell in cells:
        cell['archive'] = dict(name=cell['cell_id']+'.zip',**files[cell['cell_id']+'.zip'])
        cell['measurements_loaded'] = False
        cell['malecns_body_id'] = None
    out = dict(kind='physiology calibration record index, not fitted parameters',
        dataset_doi='10.5061/dryad.76hdr7stb',dataset_version=105831,
        source_repository='https://github.com/tony-azevedo/FlyAnalysis',source_revision=revision,
        source_record=RECORD,source_sha256=hashlib.sha256(raw).hexdigest(),
        metadata_pages=pages,cells=cells,
        counts={k:dict(indexed=sum(c['motor_class']==k for c in cells),
            main_comparison=sum(c['motor_class']==k and c['in_main_membrane_potential_comparison'] for c in cells))
            for k in ('fast','intermediate','slow')},
        limitations=['Index inclusion is not final trial acceptance or figure sample size.',
            'Raw data downloads returned HTTP 401/403; no measurements extracted.',
            'Genetic motor classes have not been registered to individual MaleCNS body IDs.',
            'No rates-to-Hz mapping, physiological fit, or behavioral success established.'])
    with a.output.open('x') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out['counts']))


if __name__ == '__main__':
    main()
