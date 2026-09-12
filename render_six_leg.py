"""Render a recorded physical pose, not an animation or behavior controller."""
import argparse
import json
from pathlib import Path
import struct
import zlib
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--trial',type=Path);source.add_argument('--poses',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();m=mujoco.MjModel.from_xml_path(str(a.xml));d=mujoco.MjData(m)
    if a.poses:
        d.qpos[:]=json.loads(a.poses.read_text())['results'][a.lane]['qpos']
    else:
        with np.load(a.trial/'trace.npz',allow_pickle=False) as t:d.qpos[:]=t['qpos'][-1,a.lane]
    d.qvel[:]=0
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.lookat[:]=d.xpos[m.body('Thorax').id];camera.distance=6;camera.azimuth=135;camera.elevation=-25
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        renderer.update_scene(d,camera);pixels=renderer.render().copy()
    def chunk(name,data):return struct.pack('!I',len(data))+name+data+struct.pack('!I',zlib.crc32(name+data)&0xffffffff)
    content=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',640,480,8,2,0,0,0))
    content+=chunk(b'IDAT',zlib.compress(b''.join(b'\x00'+row.tobytes() for row in pixels)))+chunk(b'IEND',b'')
    with a.output.open('xb') as f:f.write(content)
    print(json.dumps(dict(rendered=True,lane=a.lane,recorded_final_pose=bool(a.trial),offline_pose=bool(a.poses))))


if __name__=='__main__':main()
