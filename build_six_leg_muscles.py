"""Experimental six-leg replication of a published muscle-leg assembly.

Foreleg geometry is reused on every leg. This is not measured six-leg anatomy.
Original assets and scenes are never overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def numbers(value,default):return np.fromstring(value,sep=' ') if value else np.asarray(default,dtype=float)
def string(value):return ' '.join(format(float(x),'.17g') for x in value)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','rig-scene','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source_sha=sha(a.xml)
    if source_sha!='d67ff06d0c684599f00d5f33f8f8ac8ced01ab994c41341b551f8148b6d8aa0b':
        raise ValueError('unexpected source muscle asset')
    root=ET.parse(a.xml).getroot();source=mujoco.MjModel.from_xml_path(str(a.xml));sd=mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source,sd,0);mujoco.mj_forward(source,sd)
    rig=json.loads(a.rig_scene.read_text());rig_xml=a.rig_scene.parent/rig['scene_xml']
    assert sha(rig_xml)==rig['scene_xml_sha256']
    rm=mujoco.MjModel.from_xml_path(str(rig_xml));rd=mujoco.MjData(rm)
    mujoco.mj_resetDataKeyframe(rm,rd,0);mujoco.mj_forward(rm,rd)
    thorax=root.find('.//body[@name="Thorax"]')
    template=thorax.find('body[@name="LFCoxa"]')
    if template is None:raise ValueError('missing direct foreleg child')
    template=copy.deepcopy(template)
    old_hip=numbers(template.get('pos'),[0,0,0])
    leg_tags=['lf','lm','lh','rf','rm','rh'];mounts={}
    for tag in leg_tags:
        body=thorax.find(f'body[@name="{tag.upper()}Coxa"]')
        if body is None:raise ValueError('missing source mounting position')
        mounts[tag]=numbers(body.get('pos'),[0,0,0])
        thorax.remove(body)
    # All original equalities are right-foreleg locks. No controller survives.
    for name in ('equality','keyframe'):
        element=root.find(name)
        if element is not None:root.remove(element)
    tendons=copy.deepcopy(list(root.find('tendon')));actuators=copy.deepcopy(list(root.find('actuator')))
    root.find('tendon').clear();root.find('actuator').clear()
    thorax_sites={s.get('name'):copy.deepcopy(s) for s in thorax.findall('site')}
    required_sites={e.get('site') for t in tendons for e in t if e.tag=='site'}
    origin_sites={n:s for n,s in thorax_sites.items() if n in required_sites}
    for site in thorax.findall('site'):
        if site.get('name') in origin_sites:thorax.remove(site)
    # Preserve and authenticate mesh files. Material/texture declarations remain.
    asset=root.find('asset');assets=[];mirrored={}
    compiler=root.find('compiler');meshdir=compiler.get('meshdir','')
    (a.output/'meshes').mkdir()
    for mesh in list(asset.findall('mesh')):
        file=mesh.get('file')
        if not file:continue
        source_path=a.xml.parent/meshdir/file
        dest='meshes/'+sha(source_path)+source_path.suffix
        if not (a.output/dest).exists():shutil.copyfile(source_path,a.output/dest)
        mesh.set('file',dest);assets.append(dict(source_file=file,sha256=sha(source_path),file=dest))
        clone=copy.deepcopy(mesh);clone.set('name',mesh.get('name')+'_mirror_y')
        clone.set('scale',string(numbers(mesh.get('scale'),[1,1,1])*[1,-1,1]))
        asset.append(clone);mirrored[mesh.get('name')]=clone.get('name')
    compiler.attrib.pop('meshdir',None)
    # Reflection keeps q unchanged: rotation axes are axial vectors, -S*a.
    reflect=np.diag([1.,-1.,1.]);records=[]
    source_dir=sd.xpos[source.body('LFTarsus5').id]-sd.xpos[source.body('LFCoxa').id]
    sr=sd.xmat[source.body('Thorax').id].reshape(3,3);source_dir=sr.T@source_dir
    rr=rd.xmat[rig['residents'][0]['root_body_id']].reshape(3,3)
    prefix=rig['residents'][0]['id']+'/'
    for tag in leg_tags:
        mirror=tag.startswith('r');S=reflect if mirror else np.eye(3)
        direction=rd.xpos[rm.body(prefix+tag+'_tarsus5').id]-rd.xpos[rm.body(prefix+tag+'_coxa').id]
        direction=rr.T@direction;original=S@source_dir
        yaw=float(np.arctan2(direction[1],direction[0])-np.arctan2(original[1],original[0]))
        c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        branch=copy.deepcopy(template);name_map={e.get('name'):tag+'/'+e.get('name') for e in branch.iter() if e.get('name')}
        name_map.update({name:tag+'/'+name for name in origin_sites})
        for e in branch.iter():
            if e.get('name'):e.set('name',name_map[e.get('name')])
            if mirror:
                if e.get('pos'):e.set('pos',string(S@numbers(e.get('pos'),[0,0,0])))
                if e.get('axis'):e.set('axis',string(-S@numbers(e.get('axis'),[0,0,1])))
                if e.get('quat'):
                    mat=np.zeros(9);mujoco.mju_quat2Mat(mat,numbers(e.get('quat'),[1,0,0,0]))
                    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(S@mat.reshape(3,3)@S).ravel());e.set('quat',string(quat))
                if e.get('mesh'):e.set('mesh',mirrored[e.get('mesh')])
                if any(e.get(k) for k in ('euler','axisangle','xyaxes','zaxis','fromto')):
                    raise ValueError('unhandled reflected orientation/geometry')
        branch.set('pos',string(mounts[tag]))
        old_rotation=np.zeros(9);mujoco.mju_quat2Mat(old_rotation,numbers(branch.get('quat'),[1,0,0,0]))
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(rotation@old_rotation.reshape(3,3)).ravel());branch.set('quat',string(quat))
        thorax.append(branch)
        for name,site in origin_sites.items():
            site=copy.deepcopy(site);site.set('name',name_map[name])
            site.set('pos',string(mounts[tag]+rotation@S@(numbers(site.get('pos'),[0,0,0])-old_hip)))
            thorax.append(site)
        for tendon in tendons:
            t=copy.deepcopy(tendon);t.set('name',tag+'/'+t.get('name'))
            for e in t:
                if e.tag!='site':raise ValueError('non-site tendon path requires explicit transfer')
                e.set('site',name_map[e.get('site')])
            root.find('tendon').append(t)
        for actuator in actuators:
            act=copy.deepcopy(actuator);act.set('name',tag+'/'+act.get('name'));act.set('tendon',tag+'/'+act.get('tendon'))
            root.find('actuator').append(act)
        records.append(dict(leg=tag,source='left foreleg',mount_position=mounts[tag].tolist(),mount_yaw_radians=yaw,mirrored=mirror))
    ET.SubElement(thorax,'freejoint',name='free_thorax')
    planes=[g for g in root.findall('.//geom') if g.get('type')=='plane']
    if len(planes)!=1 or planes[0].get('name')!='floor':
        raise ValueError('expected exactly the inherited source floor')
    # Absolute mesh paths for in-memory compilation only; public XML uses relative paths.
    def compile_tree(tree):
        clone=copy.deepcopy(tree)
        for mesh in clone.findall('./asset/mesh'):
            if mesh.get('file'):mesh.set('file',str((a.output/mesh.get('file')).resolve()))
        return mujoco.MjModel.from_xml_string(ET.tostring(clone,encoding='unicode'))
    model=compile_tree(root);q=model.qpos0.copy()
    for tag in leg_tags:
        for j in range(source.njnt):
            name=source.joint(j).name
            if name.startswith('joint_LF'):
                target=model.joint(tag+'/'+name).id
                q[model.jnt_qposadr[target]]=sd.qpos[source.jnt_qposadr[j]]
    key=ET.SubElement(root,'keyframe');ET.SubElement(key,'key',name='neutral',qpos=string(q),act=string(np.zeros(model.na)))
    model=compile_tree(root);data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    if model.nu!=90 or model.nv!=48 or model.neq!=0:raise ValueError('unexpected six-leg topology')
    errors=[]
    for tag in leg_tags:
        for i in range(source.ntendon):
            target=model.tendon(tag+'/'+source.tendon(i).name).id
            errors.append(abs(float(data.ten_length[target]-sd.ten_length[i])))
    if max(errors)>1e-8:raise ValueError('muscle path lengths changed during rigid transfer')
    output=a.output/'six_leg.xml';ET.ElementTree(root).write(output,encoding='unicode')
    report=dict(kind='experimental replicated six-leg muscle body; no behavior claim',
        source_xml_sha256=source_sha,rig_scene_sha256=sha(a.rig_scene),rig_xml_sha256=sha(rig_xml),
        runner_sha256=sha(Path(__file__)),xml_sha256=sha(output),mujoco=mujoco.__version__,
        nq=model.nq,nv=model.nv,actuators=model.nu,muscle_states=model.na,equalities=model.neq,
        floor_count=int(np.count_nonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)),
        source_total_mass_model_units=float(source.body_mass.sum()),total_mass_model_units=float(model.body_mass.sum()),
        maximum_rest_tendon_length_error=max(errors),mounts=records,assets=assets,
        assumptions=['Identical foreleg assemblies on all six legs; not measured middle/hind geometry.',
                     'Right legs are mirrored left assemblies, not the original right-leg specimen.',
                     'Mount positions follow the source body; fixed horizontal leg directions follow the original whole-body rig.',
                     'Tarsal segments and other non-leg body parts remain rigid, as in the source asset.',
                     'All source muscle parameters, joint limits, damping and armature retained; no behavioral policy.'])
    with (a.output/'receipt.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ('nq','nv','actuators','equalities','maximum_rest_tendon_length_error')},indent=2))


if __name__=='__main__':main()
