"""
"""


import os
import sys
sys.path.append(os.getcwd())

import time
import math

from synth_ml.blender.wrappers import Scene, Camera, Object
from synth_ml.blender import callbacks as cb, materials
from synth_ml.objs import get_objs_folder
from synth_ml.utils.rand import seed

from vision.utils.split_data import split_data

# fence_folder = 'vision/data/fence_models/fence_with_holes'
# files = [
#     os.path.join(fence_folder, f)
#     for f in os.listdir(fence_folder)
#     if f.endswith('.obj')
# ]
files = ['vision/data/fence_models/fence_as_one.obj']

output_folder = 'vision/create_fence/data/fence'

location = (-5, 0, 0)
n_objs =1
res = 720
n_samples = 64
radius = 0.1
n_imgs = 300

seed(0)
start_time = time.time()

for i, fence_path in enumerate(files):

    s = Scene()

    cam = Camera(scene=s, location=location).look_at((0, 0, 0))

    obj = Object.import_obj(obj_path=fence_path)[0]

    objs = [obj.copy(scene=s, pass_index=i + 1) for i in range(n_objs)]
    mats = [materials.UniformMaterial() for _ in objs]

    ts = cb.TransformSampler()
    for o, mat in zip(objs, mats):
        o.assign_material(mat)
        ts.scale_uniform(obj=o, mi=15.0, ma=20.0)
        ts.pos_axis_uniform(obj=o, axis='z', mi=-0.1, ma=-2.0)
        ts.rot_axis_uniform(obj=o, axis='z', mi=math.pi/2, ma=math.pi/2)
        ts.rot_axis_uniform(obj=o, axis='x', mi=math.pi/2, ma=math.pi/2)

    s.callback = cb.CallbackCompose(
        cb.Renderer(scene=s, resolution=res, engines='CYCLES', cycles_samples=n_samples, save_image=True,
                    denoise=False, depth=True, normal=False, object_index=True, output_folder=output_folder),
        cb.MetadataLogger(scene=s, objects=objs, output_folder=output_folder),
        *[m.sampler_cb() for m in mats],
        cb.HdriEnvironment(scene=s),
        ts,
    )

    frames = range(i*n_imgs, i*n_imgs+n_imgs)
    s.render_frames(frames)

print('Splitting the data into train and validation..')
split_data(
    input_dir=f'{output_folder}/cycles',
    output_dir=output_folder,
    split_train=True
)

end_time = time.time() - start_time
print(f'Done! It took {end_time//60:.0f} min {end_time%60:.0f} sec')