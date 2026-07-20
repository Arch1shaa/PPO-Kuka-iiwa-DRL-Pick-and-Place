import pybullet as p
import pybullet_data
import numpy as np
from custom_env import CustomKukaEnv

env = CustomKukaEnv(renders=False, cameraRandom=0, isDiscrete=False, removeHeightHack=False, maxSteps=200)
env.reset()
for _ in range(10):
    p.stepSimulation()

print('picking tray uid', env._picking_tray_uid)
print('placing tray uid', env._placement_tray_uid)
print('object uids', env._objectUids)

camera_data = p.getCameraImage(width=env._width, height=env._height,
                               viewMatrix=env._view_matrix,
                               projectionMatrix=env._proj_matrix,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
print('camera_data len', len(camera_data))
if len(camera_data) == 5:
    _, _, rgb, depth, seg = camera_data
elif len(camera_data) == 4:
    _, rgb, depth, seg = camera_data
else:
    raise RuntimeError(f'Unexpected camera output format: {len(camera_data)}')

seg = np.asarray(seg, dtype=np.int32)
print('seg shape', seg.shape, 'dtype', seg.dtype)
if seg.ndim == 3:
    print('seg channels', seg.shape)
seg0 = seg if seg.ndim == 2 else seg[:, :, 0]
print('seg0 unique count', len(np.unique(seg0)))
print('seg0 unique sample', np.unique(seg0)[:50])
print('tray counts', np.bincount(seg0[seg0>=0].flatten() + 1)[:20])

mask = np.bitwise_and(seg0, 0xFFFFFF)
print('mask unique sample', np.unique(mask)[:50])
print('mask tray count', np.count_nonzero(mask==env._picking_tray_uid), np.count_nonzero(mask==env._placement_tray_uid))
print('mask object counts')
for uid in env._objectUids:
    print(uid, np.count_nonzero(mask==uid))

p.disconnect()
