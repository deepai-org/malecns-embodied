"""Create a separately hashed passive-mechanics ablation of the pinned scene.

This is a causal diagnostic, not a claim that a fly has no passive elasticity.
Geometry, muscle/servo definitions, initial state and sensory metadata stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

SCENE_SHA256 = 'accd5e86a918987ad7b39be0cba070e5ef5fa214101a0927d0f424a90edb60ef'
XML_SHA256 = 'c880facc75d6a2ac3bcbfa18ae30320f5a6110a77176268157963ba3889fbb6a'


def transform(xml_bytes, stiffness_scale, damping_scale):
    if not 0 <= stiffness_scale <= 1 or not 0 <= damping_scale <= 1:
        raise ValueError('ablation scales must be finite values in [0,1]')
    root = ET.fromstring(xml_bytes)
    changes = []
    for joint in root.findall('.//worldbody//joint'):
        if joint.get('type') == 'free':
            continue
        # Refuse to silently treat inherited joint defaults as explicit values.
        if joint.get('stiffness') is None or joint.get('damping') is None:
            raise ValueError('missing explicit passive mechanics')
        before = {key:float(joint.get(key)) for key in ('stiffness','damping')}
        after = dict(stiffness=before['stiffness']*stiffness_scale,
                     damping=before['damping']*damping_scale)
        for key,value in after.items():
            joint.set(key,format(value,'.17g'))
        changes.append(dict(joint=joint.get('name'),before=before,after=after))
    return ET.tostring(root,encoding='utf-8',xml_declaration=True),changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stiffness-scale',type=float,required=True)
    p.add_argument('--damping-scale',type=float,default=1)
    a = p.parse_args()
    source_bytes = a.scene.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=SCENE_SHA256:
        raise ValueError('unexpected source scene pin')
    scene = json.loads(source_bytes)
    xml_bytes = (a.scene.parent/scene['scene_xml']).read_bytes()
    if hashlib.sha256(xml_bytes).hexdigest()!=XML_SHA256:
        raise ValueError('unexpected source XML pin')
    modified,changes = transform(xml_bytes,a.stiffness_scale,a.damping_scale)
    if len(changes)!=504:
        raise ValueError('unexpected joint inventory')
    # Copy only authenticated assets to a fresh directory. Never edit upstream.
    for asset in scene['mesh_assets']:
        path = Path(asset['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe asset path')
        if hashlib.sha256((a.scene.parent/path).read_bytes()).hexdigest()!=asset['sha256']:
            raise ValueError('source asset digest mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    for asset in scene['mesh_assets']:
        target = a.output/asset['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.scene.parent/asset['path'],target)
    (a.output/scene['scene_xml']).write_bytes(modified)
    digest = hashlib.sha256(modified).hexdigest()
    scene['source_mjcf_sha256'] = digest
    scene['scene_xml_sha256'] = digest
    provenance = dict(kind='passive-mechanics diagnostic, not physiological calibration',
                      source_scene_sha256=SCENE_SHA256,source_xml_sha256=XML_SHA256,
                      derived_xml_sha256=digest,stiffness_scale=a.stiffness_scale,
                      damping_scale=a.damping_scale,changes=changes,
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    scene['experimental_passive_variant'] = {k:v for k,v in provenance.items() if k!='changes'}
    (a.output/'world.json').write_text(json.dumps(scene,separators=(',',':')))
    provenance['derived_scene_sha256'] = hashlib.sha256((a.output/'world.json').read_bytes()).hexdigest()
    (a.output/'passive-variant.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in provenance.items() if k!='changes'},indent=2))


if __name__ == '__main__':
    main()
