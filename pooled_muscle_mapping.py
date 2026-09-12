"""Coarse named motor-pool recruitment for all published foreleg muscle units.

Subdivision sharing is a hypothesis, not measured neuromuscular connectivity.
"""
import argparse
import hashlib
import json
from pathlib import Path

POOLS = {
    'LFC_tergopleural_promotor_a': ['Tergopleural/Pleural promotor MN'],
    'LFC_tergopleural_promotor_b': ['Tergopleural/Pleural promotor MN'],
    'LFC_pleural_promotor': ['Tergopleural/Pleural promotor MN'],
    'LFC_pleural_remotor_and_abductor': ['Pleural remotor/abductor MN'],
    'LFC_sternal_anterior_rotator': ['Sternal anterior rotator MN'],
    'LFC_sternal_posterior_rotator': ['Sternal posterior rotator MN'],
    'LFC_sternal_adductor': ['Sternal adductor MN'],
    'LFF_trochanter_flexor_a': ['Tr flexor MN'],
    'LFF_trochanter_flexor_b': ['Tr flexor MN'],
    'LFF_sterno-tergo-trochanter_extensor_a': ['Sternotrochanter MN', 'Tergotr. MN'],
    'LFF_sterno-tergo-trochanter_extensor_b': ['Sternotrochanter MN', 'Tergotr. MN'],
    'LFF_accesory_trochanter_flexor': ['Acc. tr flexor MN'],
    'LFF_trochanter_extensor': ['Tr extensor MN'],
    'LFTibia_flex_93434': ['Ti flexor MN'],
    'LFTibia_extensor_93932': ['Ti extensor MN'],
}


def make_mapping(inventory, annotation):
    if inventory['neuron_source_sha256'] != annotation['source_sha256']:
        raise ValueError('annotation identity mismatch')
    if len(inventory['muscles']) != 15 or {m['muscle'] for m in inventory['muscles']} != set(POOLS):
        raise ValueError('unexpected muscle inventory')
    muscles=[]
    for m in inventory['muscles']:
        labels=POOLS[m['muscle']]
        records=[r for r in annotation['motors'] if r['mancType'] in labels and
                 r['subclass']=='fl' and (r['somaSide'] or r['rootSide'])=='L']
        if {r['mancType'] for r in records} != set(labels):
            raise ValueError('missing candidate motor pool')
        muscles.append(dict(muscle=m['muscle'],muscle_id=m['muscle_id'],candidate_motor_rows=[r['cns_row'] for r in records],
            candidate_body_ids=[r['bodyId'] for r in records],annotation_labels=labels,
            mapping_status='shared named-pool approximation; no subdivision-specific wiring claim'))
    return dict(kind='candidate pooled recruitment; not a mechanical trial',actuators=15,
        xml_sha256=inventory['xml_sha256'],neuron_source_sha256=annotation['source_sha256'],muscles=muscles,
        assumptions=['All modeled subdivisions with the same pool receive its mean normalized rate.',
                     'Composite sterno/tergo units share the union of the two named pools.',
                     'No motor-unit-specific recruitment thresholds, strengths or subdivision assignments are known.',
                     'No muscle force capacity is changed; underlying geometry is still incomplete foreleg-only anatomy.',
                     'Accessory tibia flexor and other muscles absent from this asset remain unmodeled.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','annotation','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=make_mapping(json.loads(a.inventory.read_text()),json.loads(a.annotation.read_text()))
    result['input_sha256']={name:hashlib.sha256(getattr(a,name).read_bytes()).hexdigest() for name in ('inventory','annotation')}
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(muscles=len(result['muscles']),unique_neurons=len({r for m in result['muscles'] for r in m['candidate_motor_rows']}))))


if __name__=='__main__':main()
