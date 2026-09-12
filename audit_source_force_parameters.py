"""Compare published OpenSim/MuJoCo parameters without interpreting their units."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('opensim','mujoco','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    # Fixed source hash, not a search for a more favorable source version.
    assert sha(a.mujoco)=='d67ff06d0c684599f00d5f33f8f8ac8ced01ab994c41341b551f8148b6d8aa0b'
    source=ET.parse(a.opensim).getroot();target=ET.parse(a.mujoco).getroot()
    muscles={m.get('name'):m for m in source.findall('.//Millard2012EquilibriumMuscle')}
    results=[]
    for actuator in target.findall('./actuator/general'):
        original=muscles[actuator.get('name')]
        gain=list(map(float,actuator.get('gainprm').split()))
        force=float(original.findtext('max_isometric_force'))
        results.append(dict(name=actuator.get('name'),opensim_fmax=force,
            mujoco_fmax=gain[2],ratio=gain[2]/force,
            opensim_optimal_fiber_length=float(original.findtext('optimal_fiber_length')),
            opensim_tendon_slack_length=float(original.findtext('tendon_slack_length')),
            mujoco_gainprm=gain,mujoco_dynprm=list(map(float,actuator.get('dynprm').split()))))
    report=dict(kind=__doc__,runner_sha256=sha(Path(__file__)),
        inputs={name:sha(getattr(a,name)) for name in ('opensim','mujoco')},
        opensim_declared_length_units=source.findtext('.//length_units'),
        opensim_declared_force_units=source.findtext('.//force_units'),
        opensim_gravity=source.findtext('.//gravity'),
        mujoco_gravity=target.find('option').get('gravity'),muscles=results,
        caveats=['Both are published artifacts, not an independently rerun conversion.',
                 'Ratios vary by muscle; these parameters do not establish a single omitted conversion multiplier.',
                 'OpenSim unit labels alone do not validate physical SI interpretation of the scaled model.',
                 'No source asset, muscle parameter or runtime setting is changed.'])
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(muscles=len(results),minimum_ratio=min(r['ratio'] for r in results),
                         maximum_ratio=max(r['ratio'] for r in results))))


if __name__=='__main__':main()
