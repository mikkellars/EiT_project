"""
Create synthetic images of fences.

Command line:
    synth-ml-blender create_data.py --compute-devices 0

Arguments:
    obj_filename = 'data/fence_models/fence.ply'
    number objects = 5
    camera distance = -0.2
    sphere object radius = 0.1
    image resolution = 512
    cycles samples = 32
    number images = 100
"""


import random
import time
import datetime
import math
from synth_ml.blender.wrappers import Scene, Camera, Object
from synth_ml.blender import callbacks as cb, materials
from synth_ml.objs import get_objs_folder
from synth_ml.utils.rand import seed


obj_filename = 'data/fence_models/fence.ply'
n_objs = 5
cam_dist = -0.2
sphere_obj_rad = 0.1
im_res = 512
cycles_samples = 32
n_ims = 100


def main():

    # -----
    # Setup
    # -----

    print('Setting up..')

    seed(0)
    s = Scene()

    cam = Camera(scene=s, location=(cam_dist, 0, 0)).look_at((0, 0, 0))
    obj = Object.import_obj(obj_path=obj_filename)[0]
    objs = [obj.copy(scene=s, pass_index=i + 1) for i in range(n_objs)]
    mats = [materials.UniformMaterial() for _ in objs]

    ts = cb.TransformSampler()
    for o, mat in zip(objs, mats):
        o.assign_material(mat)
        ts.pos_sphere_volume(o, center=[0, 0, 0], radius=sphere_obj_rad)
        ts.rot_uniform(o)

    s.callback = cb.CallbackCompose(
        cb.Renderer(scene=s, resolution=im_res, engines='CYCLES', cycles_samples=cycles_samples, object_index=True),
        cb.MetadataLogger(scene=s, objects=objs),
        *[m.sampler_cb(p_metallic=0.8) for m in mats],
        cb.HdriEnvironment(scene=s),
        ts,
    )

    start_render_time = time.time()

    s.render_frames(range(n_ims))

    print(f'Rendering took {datetime.timedelta(seconds=math.ceil(time.time() - start_render_time))}')


if __name__ == "__main__":
    print(__doc__)
    main()
