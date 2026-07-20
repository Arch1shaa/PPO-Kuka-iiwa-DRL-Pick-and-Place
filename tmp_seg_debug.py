import pybullet as p
import pybullet_data
import numpy as np
from custom_env import CustomKukaEnv

env = CustomKukaEnv(renders=False, cameraRandom=0, isDiscrete=False, removeHeightHack=False, maxSteps=200)
env.reset()
for _ in range(20):
    p.stepSimulation()

print('picking tray uid', env._picking_tray_uid)
print('placing tray uid', env._placement_tray_uid)
print('object uids', env._objectUids)
print('camera view', env._view_matrix[:4], env._proj_matrix[:4])

cam = p.getCameraImage(width=env._width, height=env._height,
                       viewMatrix=env._view_matrix, projectionMatrix=env._proj_matrix,
                       flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
print('camera tuple len', len(cam))
for i, item in enumerate(cam):
    if isinstance(item, np.ndarray):
        print(i, 'ndarray', item.shape, item.dtype, item.size)
    else:
        print(i, type(item))

_, _, rgb, depth, seg = cam
seg = np.asarray(seg, dtype=np.int32)
print('seg initial ndim', seg.ndim, 'shape', seg.shape)
if seg.ndim == 1 and seg.size == env._height * env._width:
    seg = seg.reshape((env._height, env._width))
print('seg reshaped ndim', seg.ndim, 'shape', seg.shape)
print('seg unique count', len(np.unique(seg)))
print('seg unique sample', np.unique(seg)[:100])

mask = np.bitwise_and(seg, 0xFFFFFF)
print('mask unique count', len(np.unique(mask)))
print('mask unique sample', np.unique(mask)[:100])
print('tray counts', np.count_nonzero(mask == env._picking_tray_uid), np.count_nonzero(mask == env._placement_tray_uid))
for uid in env._objectUids:
    print('obj', uid, np.count_nonzero(mask == uid))

# Print a few mask values near center
h, w = env._height, env._width
print('center sample', mask[h//2-2:h//2+3, w//2-2:w//2+3])

p.disconnect()