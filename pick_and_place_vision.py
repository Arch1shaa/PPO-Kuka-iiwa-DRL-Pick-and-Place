import os
import time
import numpy as np
import pybullet as p
from voice_robot_arm.voice_command import get_voice_command

os.environ["KUKA_SKIP_GLOBAL_ENV"] = "1"

from custom_env import CustomKukaEnv


def acquire_camera_frame(env):
    camera_data = p.getCameraImage(width=env._width,
                                  height=env._height,
                                  viewMatrix=env._view_matrix,
                                  projectionMatrix=env._proj_matrix,
                                  flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    if len(camera_data) == 5:
        _, _, rgb, depth, seg = camera_data
    elif len(camera_data) == 4:
        _, rgb, depth, seg = camera_data
    else:
        raise RuntimeError(f'Unexpected camera output format: {len(camera_data)}')

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 1 and rgb.size == env._height * env._width * 4:
        rgb = rgb.reshape((env._height, env._width, 4))
    rgb = rgb[:, :, :3]

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 1 and depth.size == env._height * env._width:
        depth = depth.reshape((env._height, env._width))

    seg = np.asarray(seg, dtype=np.int32)
    if seg.ndim == 1 and seg.size == env._height * env._width:
        seg = seg.reshape((env._height, env._width))
    if seg.ndim == 3 and seg.shape[2] == 4:
        seg = seg[:, :, 0]
    return rgb, depth, seg


def decode_segmentation_mask(seg):
    seg = np.asarray(seg, dtype=np.int32)
    if seg.ndim == 3 and seg.shape[2] == 4:
        seg = seg[:, :, 0]
    uid_mask = np.bitwise_and(seg, 0xFFFFFF)
    uid_mask[seg < 0] = -1
    return uid_mask


def find_tray_masks(env, uid_mask):
    tray_ids = [env._picking_tray_uid, env._placement_tray_uid]
    masks = {}
    for tray_id in tray_ids:
        masks[tray_id] = uid_mask == tray_id
    return masks


def tray_bounding_box(tray_mask):
    indices = np.column_stack(np.nonzero(tray_mask))
    if indices.size == 0:
        return None
    min_y, min_x = indices.min(axis=0)
    max_y, max_x = indices.max(axis=0)
    return min_y, min_x, max_y, max_x


def detect_objects_in_tray(env, uid_mask, tray_mask):
    source_objects = []
    for uid in env._objectUids:
        obj_mask = uid_mask == uid
        if not np.any(obj_mask):
            continue
        pixel_indices = np.column_stack(np.nonzero(obj_mask))
        centroid_px = pixel_indices.mean(axis=0)
        cy, cx = int(centroid_px[0]), int(centroid_px[1])
        pos, orn = p.getBasePositionAndOrientation(uid)
        if not env._is_object_in_picking_tray(pos):
            continue
        euler = p.getEulerFromQuaternion(orn)
        grasp_point = [pos[0], pos[1], pos[2] + 0.02]
        source_objects.append({
            'uid': uid,
            'pixel_area': int(np.count_nonzero(obj_mask)),
            'centroid': (cx, cy),
            'orientation': tuple(float(v) for v in euler),
            'grasp_point': tuple(grasp_point),
            'world_position': tuple(pos),
        })
    return source_objects


def classify_trays(env, uid_mask):
    tray_masks = find_tray_masks(env, uid_mask)
    if not any(np.any(mask) for mask in tray_masks.values()):
        raise RuntimeError('Unable to detect trays from camera image')
    source_tray = env._picking_tray_uid
    dest_tray = env._placement_tray_uid
    return source_tray, dest_tray, tray_masks


def select_target_object(source_objects):
    if not source_objects:
        return None
    if len(source_objects) == 1:
        return source_objects[0]['uid']

    def clearance_score(item):
        x, y, _ = item['world_position']
        distances = []
        for other in source_objects:
            if other['uid'] == item['uid']:
                continue
            ox, oy, _ = other['world_position']
            distances.append(float(np.hypot(x - ox, y - oy)))
        return min(distances) if distances else 0.0

    return max(source_objects, key=clearance_score)['uid']


def verify_transfer(env, target_uid, uid_mask, tray_masks):
    obj_mask = uid_mask == target_uid
    if np.any(obj_mask):
        source_overlap = np.count_nonzero(obj_mask & tray_masks[env._picking_tray_uid])
        dest_overlap = np.count_nonzero(obj_mask & tray_masks[env._placement_tray_uid])
        in_source = int(source_overlap > 0)
        in_dest = int(dest_overlap > 0)
    else:
        pos, _ = p.getBasePositionAndOrientation(target_uid)
        in_source = int(abs(pos[0] - env._picking_tray_center[0]) < 0.22 and
                        abs(pos[1] - env._picking_tray_center[1]) < 0.18 and
                        pos[2] > env._picking_tray_center[2])
        in_dest = int(env._is_object_in_placing_tray(pos))
    success = (in_source == 0 and in_dest > 0)
    return success, in_source, in_dest


def get_source_objects_from_current_frame(env):
    rgb, depth, seg = acquire_camera_frame(env)
    uid_mask = decode_segmentation_mask(seg)
    source_tray, dest_tray, tray_masks = classify_trays(env, uid_mask)
    source_objects = detect_objects_in_tray(env, uid_mask, tray_masks[source_tray])
    return source_objects, uid_mask, tray_masks
def get_target_uid_by_name(env, object_name):

    color_map = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "purple": 4,
    }

    object_name = object_name.lower().strip()

    if object_name not in color_map:
        return None

    index = color_map[object_name]

    if index >= len(env._objectUids):
        return None

    return env._objectUids[index]

def pick_and_place_loop(max_cycles=100):
    print('Opening PyBullet GUI')
    env = CustomKukaEnv(renders=True, cameraRandom=0, isDiscrete=False, removeHeightHack=False, maxSteps=200)
    print('Resetting environment')
    env.reset()
    print("\nOBJECT LIST")

    for i, uid in enumerate(env._objectUids):
      pos, _ = p.getBasePositionAndOrientation(uid)

      print(
        f"index={i} uid={uid} pos={pos}"
    )
    time.sleep(0.1)

    cycles = 0
    '''target_name = input(
    "\nEnter object to pick "
    "(red/green/blue/yellow/purple): "
    )
    target_uid = env.get_uid_by_color(
        target_name
        )
    if target_uid is None:
        print("Invalid object name")
        return'''
    while True:
        print("\nspeak to pick up ye object")
        target_name= get_voice_command()
        if target_name is None:
            print("no valid color detected.")
            continue
       # target_name = input(
        #    "\nEnter object to pick "
         #   "(red/green/blue/yellow/purple): "
          #  ).lower().strip()
        target_uid = env.get_uid_by_color(
            target_name
            )
        if target_uid is None:
            print(
                f"Object '{target_name}' does not exist."
                )
            continue
        pos, _ = p.getBasePositionAndOrientation(
            target_uid
            )
        if env._is_object_in_placing_tray(pos):
            print(
                f"{target_name} is already in "
                f"the destination tray."
                )
            continue
        break
    print(
        f"Selected {target_name} "
        f"(uid={target_uid})"
        )
    while cycles < max_cycles:
        try:
            source_objects, uid_mask, tray_masks = get_source_objects_from_current_frame(env)
        except RuntimeError as error:
            print(f'Perception failure: {error}. Retrying frame.')
            time.sleep(0.2)
            continue

        object_count = len(source_objects)
        if object_count==0:
            print("\nNo more objects left in source tray.")
            print("Simulation Complete.")
            break
        print(f'Cycle {cycles}: Objects detected in source tray: {object_count}')

        if env._all_objects_placed():
            print('Transfer Complete')
            break
        if object_count == 0:
            print('No valid source objects detected, but transfer is not complete. Retrying perception.')
            time.sleep(0.2)
            cycles += 1
            continue

        if target_uid is not None and target_uid not in env._objectUids:
            print(
                f"{target_name} already moved "
                f"or not available."
                )
            break
        print(
            f"Picking requested object "
            f"{target_name} "
            f"(uid={target_uid})"
            )
        '''if target_uid is None:
            target_uid = select_target_object(
                source_objects
                )
            print(
                f"Automatically selected uid "
                f"{target_uid}"
                )
        else:
            print(
                f"Picking requested object "
                f"{target_name} "
                f"(uid={target_uid})"
                )'''

        try:
            pos, _ = p.getBasePositionAndOrientation(target_uid)
            if not env._is_object_in_picking_tray(pos):
                print(
                    f"{target_name} is not in the source tray."
                    )
                while True:
                    print("\nSpeak another object...")
                    target_name = get_voice_command()#input(
                       # "\nChoose another object: "
                        #).lower().strip()
                    target_uid = env.get_uid_by_color(
                        target_name
                        )
                    if target_uid is None:
                        print("Object does not exist.")
                        continue
                    pos, _ = p.getBasePositionAndOrientation(
                        target_uid
                        )
                    if not env._is_object_in_picking_tray(pos):
                        print(
                            f"{target_name} is already in destination tray."
                            )
                        continue
                    break
            print(f'Picking and placing object {target_uid}...')
            env.pick_and_place_object(target_uid)
            env._settle_simulation(150)
            print(f'Pick-and-place completed for uid {target_uid}')
        except Exception as e:
            print(f'Pick-and-place failed: {e}')
            import traceback
            traceback.print_exc()
            print('Retrying pick on a fresh perception frame')
            time.sleep(0.1)
            cycles += 1
            continue

        time.sleep(0.1)
        try:
            _, uid_mask, tray_masks = get_source_objects_from_current_frame(env)
            success, in_source, in_dest = verify_transfer(env, target_uid, uid_mask, tray_masks)
        except RuntimeError as error:
            print(f'Post-placement perception failed: {error}. Falling back to world check.')
            pos, _ = p.getBasePositionAndOrientation(target_uid)
            success = not (abs(pos[0] - env._picking_tray_center[0]) < 0.22 and
                           abs(pos[1] - env._picking_tray_center[1]) < 0.18 and
                           pos[2] > env._picking_tray_center[2]) and env._is_object_in_placing_tray(pos)
            in_source = 0 if success else 1
            in_dest = 1 if success else 0

        print(f'Verification: uid {target_uid}: in_source={in_source}, in_destination={in_dest}, transfer_ok={success}')

        if not success:
            pos, _ = p.getBasePositionAndOrientation(target_uid)
            print(f"placement failed.object world position={pos}")
            cycles+=1
            continue
         #   print('Placement verification failed; retrying next cycle')
          #  cycles += 1
           # continue
        print(
        f"{target_name} transferred successfully."
        )
        cycles += 1
        #target_uid=None
        while True:
            print("Returning to home position...")
            home_position = [
                env._picking_tray_center[0],
                env._picking_tray_center[1],
                0.60
                ]
            env._move_gripper_to(
                home_position,
                finger_angle=0.30,
                steps=150
                )
            print("Ready for next object.")
            print("\nSpeak to select the object...")
            target_name = get_voice_command()#input(
                #"\nEnter next object "
                #"(red/green/blue/yellow/purple): "
                #).lower().strip()
            if target_name is None:
                print("no valid color detected.")
                continue
            target_uid = env.get_uid_by_color(
                target_name
                )
            if target_uid is None:
                print(
                    f"Object '{target_name}' does not exist."
                    )
                continue
            pos, _ = p.getBasePositionAndOrientation(
                target_uid
                )
            if env._is_object_in_placing_tray(pos):
                print(
                    f"{target_name} is already in the destination tray."
                    )
                continue
            print(
                f"Selected {target_name} "
                f"(uid={target_uid})"
                )
            break
            #f"(uid={target_uid})"
            #)

    if env._all_objects_placed():
        print(f'Done; total cycles: {cycles}')
    else:
        print(f'Stopped after {cycles} cycles before all objects reached the destination tray.')
    p.disconnect()
    return




if __name__ == '__main__':
    pick_and_place_loop(max_cycles=200)
