import os
import sys
import time
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.optim as optim

# Import and patch gym for compatibility
import gym
from collections import UserDict

registry = UserDict(gym.envs.registration.registry)
if not hasattr(registry, "env_specs"):  # Compatibility fix for newer Gym versions
    registry.env_specs = registry
gym.envs.registration.registry = registry

# Import PyBullet
import pybullet_envs
import pybullet as p
import pybullet_data
from pybullet_envs.bullet import kuka
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

# Initialize the Kuka Diverse Object Environment
class CustomKukaEnv(KukaDiverseObjectEnv):
    def get_uid_by_color(self, color_name):
        color_name = color_name.lower()
        color_lookup = {
            "red":    [1, 0, 0, 1],
            "green":  [0, 1, 0, 1],
            "blue":   [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1],
            }
        if color_name not in color_lookup:
            return None
        target_color = color_lookup[color_name]
        for uid, obj in zip(
            self._objectUids,
            self._fixed_objects
            ):
            if obj["rgbaColor"] == target_color:
                return uid
        return None
        
    def _gripper_xy_error(self, target_xy):
     state = p.getLinkState(self._kuka.kukaUid,
                           self._kuka.kukaEndEffectorIndex)

     ee_pos = state[0]

     return np.hypot(
        ee_pos[0] - target_xy[0],
        ee_pos[1] - target_xy[1]
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graspSuccess = 0
        self._attempted_grasp = False  # Track whether a grasp has been attempted.
        # Fixed objects start inside the picking tray area.
        # The built-in robot tray remains as the picking tray.
        '''self._fixed_objects = [
            {
                "shape": "cube",
                "halfExtents": [0.03, 0.03, 0.03],
                "rgbaColor": [1, 0, 0, 1],
                "position": [0.62, 0.10, 0.05],
                "mass": 0.35,
            },
            {
                "shape": "sphere",
                "radius": 0.035,
                "rgbaColor": [0, 1, 0, 1],
                "position": [0.60, 0.06, 0.05],
                "mass": 0.25,
            },
            {
                "shape": "cube",
                "halfExtents": [0.025, 0.025, 0.025],
                "rgbaColor": [0, 0, 1, 1],
                "position": [0.58, 0.04, 0.05],
                "mass": 0.20,
            },
            {
                "shape": "sphere",
                "radius": 0.032,
                "rgbaColor": [1, 1, 0, 1],
                "position": [0.61, 0.14, 0.05],
                "mass": 0.20,
            },
            {
                "shape": "cube",
                "halfExtents": [0.028, 0.028, 0.04],
                "rgbaColor": [1, 0, 1, 1],
                "position": [0.56, 0.03, 0.06],
                "mass": 0.30,
            },
        ]'''
        self._fixed_objects = [
    {
        "shape": "cube",
        "halfExtents": [0.03, 0.03, 0.03],
        "rgbaColor": [1, 0, 0, 1],
        "position": [0.56, 0.05, 0.05],
        "mass": 0.35,
    },
    {
        "shape": "sphere",
        "radius": 0.035,
        "rgbaColor": [0, 1, 0, 1],
        "position": [0.64, 0.00, 0.05],
        "mass": 0.25,
    },
    {
        "shape": "cube",
        "halfExtents": [0.025, 0.025, 0.025],
        "rgbaColor": [0, 0, 1, 1],
        "position": [0.56, 0.11, 0.05],
        "mass": 0.20,
    },
    {
        "shape": "sphere",
        "radius": 0.032,
        "rgbaColor": [1, 1, 0, 1],
        "position": [0.62, 0.15, 0.10],
        "mass": 0.20,
    },
    {
        "shape": "cube",
        "halfExtents": [0.028, 0.028, 0.04],
        "rgbaColor": [1, 0, 1, 1],
        "position": [0.60, 0.08, 0.06],
        "mass": 0.30,
    },
]
        self._placement_tray_position = [0.58, -0.30, -0.19]

    def reset(self):
        """Reset the environment with fixed objects and a relocated tray."""
        # Set camera parameters.
        look = [0.23, 0.2, 0.54]
        distance = 1.0
        pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
        yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20.0 + self._cameraRandom * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10.0
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -0.820000,
                   0.000000, 0.000000, 0.0, 1.0)

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._configure_gripper_motion()

        # Remove the built-in tray, then create only two trays: one picking tray and one placing tray.
        if hasattr(self._kuka, 'trayUid'):
            try:
                p.removeBody(self._kuka.trayUid)
            except Exception:
                pass

        self._picking_tray_center = [0.60, 0.08, -0.19]
        self._placement_tray_center = [0.58, -0.35, -0.19]

        self._picking_tray_uid = self._create_tray(center=self._picking_tray_center,
                                                   width=0.45,
                                                   depth=0.40,
                                                   height=0.18,
                                                   wall_thickness=0.02,
                                                   color=[0.5, 0.5, 0.5, 1])

        self._placement_tray_uid = self._create_tray(center=self._placement_tray_center,
                                                     width=0.45,
                                                     depth=0.40,
                                                     height=0.18,
                                                     wall_thickness=0.02,
                                                     color=[0.3, 0.3, 0.3, 1])

        self._objectUids = self._place_fixed_objects()
        self._configure_grasp_physics()
        self._observation = self._get_observation()
        return np.array(self._observation)

    def _configure_gripper_motion(self):
        self._kuka.maxVelocity = 0.65
        self._kuka.fingerAForce = 25
        self._kuka.fingerBForce = 25
        self._kuka.fingerTipForce = 40

    def _configure_grasp_physics(self):
        for uid in self._objectUids:
            p.changeDynamics(uid, -1,
                             lateralFriction=2.2,
                             spinningFriction=0.02,
                             rollingFriction=0.02,
                             restitution=0.0,
                             contactStiffness=30000,
                             contactDamping=1000)
            for link_id in [6, 7, 8, 10, 11, 13]:
                p.setCollisionFilterPair(self._kuka.kukaUid, uid, link_id, -1, enableCollision=1)
        for link_id in [8, 10, 11, 13]:
            p.changeDynamics(self._kuka.kukaUid, link_id,
                             lateralFriction=4.0,
                             spinningFriction=0.03,
                             rollingFriction=0.03,
                             restitution=0.0,
                             contactStiffness=30000,
                             contactDamping=1000)

    def _place_fixed_objects(self):
        object_uids = []
        for obj in self._fixed_objects:
            if obj["shape"] == "cube":
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj["halfExtents"])
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=obj["halfExtents"], rgbaColor=obj["rgbaColor"])
            else:
                col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=obj["radius"])
                vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=obj["radius"], rgbaColor=obj["rgbaColor"])

            uid = p.createMultiBody(baseMass=obj["mass"],
                                    baseCollisionShapeIndex=col_shape,
                                    baseVisualShapeIndex=vis_shape,
                                    basePosition=obj["position"],
                                    baseOrientation=[0, 0, 0, 1])
            object_uids.append(uid)
            for _ in range(240):
                p.stepSimulation()
        return object_uids

    def _create_tray(self, center, width, depth, height, wall_thickness, color):
        half_width = width / 2.0
        half_depth = depth / 2.0
        wall_height = height

        base_half = [half_width, half_depth, wall_thickness / 2.0]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=color)

        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_frame_positions = []
        link_inertial_frame_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []

        wall_specs = [
            ([0, half_depth - wall_thickness / 2, wall_height / 2], [half_width, wall_thickness / 2.0, wall_height / 2.0]),
            ([0, -half_depth + wall_thickness / 2, wall_height / 2], [half_width, wall_thickness / 2.0, wall_height / 2.0]),
            ([-half_width + wall_thickness / 2, 0, wall_height / 2], [wall_thickness / 2.0, half_depth, wall_height / 2.0]),
            ([half_width - wall_thickness / 2, 0, wall_height / 2], [wall_thickness / 2.0, half_depth, wall_height / 2.0]),
        ]
        for pos, half_extents in wall_specs:
            link_collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents))
            link_visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color))
            link_positions.append(pos)
            link_orientations.append([0, 0, 0, 1])
            link_inertial_frame_positions.append([0, 0, 0])
            link_inertial_frame_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(0)
            link_joint_types.append(p.JOINT_FIXED)
            link_joint_axes.append([0, 0, 0])

        tray_uid = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=base_col,
                                     baseVisualShapeIndex=base_vis,
                                     basePosition=[center[0], center[1], center[2] + wall_thickness / 2.0],
                                     baseOrientation=[0, 0, 0, 1],
                                     linkMasses=[0] * len(link_collision_shapes),
                                     linkCollisionShapeIndices=link_collision_shapes,
                                     linkVisualShapeIndices=link_visual_shapes,
                                     linkPositions=link_positions,
                                     linkOrientations=link_orientations,
                                     linkInertialFramePositions=link_inertial_frame_positions,
                                     linkInertialFrameOrientations=link_inertial_frame_orientations,
                                     linkParentIndices=link_parent_indices,
                                     linkJointTypes=link_joint_types,
                                     linkJointAxis=link_joint_axes)
        return tray_uid

    def step(self, action):
        """Support default 3D control plus optional 5D gripper control."""
        action = np.array(action, dtype=np.float32).flatten()
        if action.ndim == 0:
            action = action.reshape(1)

        if self._isDiscrete:
            return super().step(int(action.item()))

        if action.shape[0] == 3 and not self._removeHeightHack:
            return super().step(action)

        if action.shape[0] == 4 and self._removeHeightHack:
            return super().step(action)

        return self._step_custom_action(action)

    def _step_custom_action(self, action):
        dv = self._dv
        if action.shape[0] == 5:
            dx = dv * action[0]
            dy = dv * action[1]
            if self._removeHeightHack:
                dz = dv * action[2]
                da = 0.25 * action[3]
            else:
                dz = -dv
                da = 0.25 * action[2]
            finger = float(np.clip(action[4], 0.0, 1.0))
        elif action.shape[0] == 4 and self._removeHeightHack:
            dx = dv * action[0]
            dy = dv * action[1]
            dz = dv * action[2]
            da = 0.25 * action[3]
            finger = 0.3
        elif action.shape[0] == 4:
            dx = dv * action[0]
            dy = dv * action[1]
            dz = -dv
            da = 0.25 * action[2]
            finger = 0.3
        else:
            raise ValueError(f"Unexpected action shape for custom step: {action.shape}")

        return self._step_continuous([dx, dy, dz, da, finger])

    def _reward(self):
        """Calculates the reward based only on the placing tray."""
        reward = 0.0
        self._graspSuccess = 0

        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            if self._is_object_in_placing_tray(pos):
                self._graspSuccess += 1
                reward += 1.0

        return round(float(reward), 3)

    def _is_object_in_placing_tray(self, object_pos):
        tray_pos, _ = p.getBasePositionAndOrientation(self._placement_tray_uid)
        dx = abs(object_pos[0] - tray_pos[0])
        dy = abs(object_pos[1] - tray_pos[1])
        tray_floor_z = tray_pos[2] + 0.02
        return dx < 0.20 and dy < 0.18 and object_pos[2] > tray_floor_z

    def get_camera_image(self):
        """Return the RGB camera observation for visual policy input."""
        return self._get_observation()

    def get_object_positions(self):
        """Return the current base positions of all tracked objects."""
        return [p.getBasePositionAndOrientation(uid)[0] for uid in self._objectUids]

    def get_objects_in_picking_tray(self):
        """Return object positions still in the picking tray."""
        picking = []
        for uid in self._objectUids:
            pos = p.getBasePositionAndOrientation(uid)[0]
            if self._is_object_in_picking_tray(pos):
                picking.append(pos)
        return picking

    def _is_object_in_picking_tray(self, object_pos):
        dx = abs(object_pos[0] - self._picking_tray_center[0])
        dy = abs(object_pos[1] - self._picking_tray_center[1])
        tray_floor_z = self._picking_tray_center[2] + 0.02
        return dx < 0.20 and dy < 0.18 and object_pos[2] > tray_floor_z

    def _all_objects_placed(self):
        return all(self._is_object_in_placing_tray(p.getBasePositionAndOrientation(uid)[0])
                   for uid in self._objectUids)

    def _gripper_holding_object(self):
        """Determines if the gripper is holding an object."""
        contact_points = p.getContactPoints(self._kuka.kukaUid)
        return len(contact_points) > 0

    def _is_object_held(self, uid):
        return self._grasp_confirmed(uid, closed_finger_angle=0.0)

    def _termination(self):
        """Terminate when all objects are transferred or max steps reached."""
        return self._all_objects_placed() or self._env_step >= self._maxSteps

    def auto_transfer_all_objects(self):
        """Pick each object from the picking tray and place it in the placing tray."""
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            if self._is_object_in_placing_tray(pos):
                continue
            self.pick_and_place_object(uid)

    def pick_and_place_object(self, uid):
        print(f"Starting pick and place for uid {uid}")

        if not self.pick_object(uid):
            raise RuntimeError(f'Grasp failed for uid {uid}')

        if not self.place_object(uid):
            raise RuntimeError(f'Placement failed for uid {uid}')

        print(f"Completed pick and place for uid {uid}")
   
    def _log_task_state(
        self,
        state,
        uid=None,
        gripper_command=None,
        grasp_success=None,
        place_success=None,
        reason=None):

      print(
        f"[{state}] "
        f"uid={uid} "
        f"gripper={gripper_command} "
        f"grasp={grasp_success} "
        f"place={place_success}"
        f"reason={reason}"
    )

    def pick_object(self, uid, max_attempts=2):
        target_pos, _ = p.getBasePositionAndOrientation(uid)
       # dx = abs(target_pos[0] - self._picking_tray_center[0])
        #dy = abs(target_pos[1] - self._picking_tray_center[1])
        #if dx > 0.16 or dy > 0.12:
         #   print("Object too close to tray wall.")
          #  return False
        aabb_min, aabb_max = p.getAABB(uid)
        object_center = list(target_pos)
        object_center[0] -= 0.04
        object_center[1] -= 0.03
        #object_center=[
            #(aabb_min[0] + aabb_max[0]) / 2.0,
            #(aabb_min[1] + aabb_max[1]) / 2.0,
            #(aabb_min[2] + aabb_max[2]) / 2.0,
        #]
        object_top = aabb_max[2]
        
        open_finger = 0.30
        closed_finger = 0.0
        approach = [object_center[0], object_center[1], 0.60]#0.45
        pre_grasp = [object_center[0], object_center[1], 0.20]#0.15
        #lower = [object_center[0], object_center[1], 0.02]
        lower = [
            object_center[0],
            object_center[1],
            object_top+0.015
            ]

        for attempt in range(max_attempts):
            if attempt:
                print(f'  Retrying grasp for uid {uid}...')
                target_pos, _ = p.getBasePositionAndOrientation(uid)
                print("\nTARGET OBJECT")
                print("uid =", uid)
                print("world pos =", target_pos)
                aabb_min, aabb_max = p.getAABB(uid)
                object_center = list(target_pos)  
                object_center[0] -= 0.04
                object_center[1] -= 0.03 
                #object_center=[
                 #   (aabb_min[0] + aabb_max[0]) / 2.0,
                  #  (aabb_min[1] + aabb_max[1]) / 2.0,
                   # (aabb_min[2] + aabb_max[2]) / 2.0,
                #]
                object_top = aabb_max[2]
                approach = [object_center[0], object_center[1], object_top + 0.24]
                pre_grasp = [object_center[0], object_center[1], 0.15]
                #lower = [object_center[0], object_center[1], object_top + 0.02]
                lower = [
                    object_center[0],
                    object_center[1],
                    object_top+0.015
                    ]
            print("TARGET POS =", target_pos)
            print("OBJECT CENTER =", object_center)
            print("APPROACH =", approach)
            print("PRE_GRASP =", pre_grasp)
            print('  Approaching object...')
            hover = [
                 object_center[0],
                 object_center[1],
                 object_top + 0.30
                 ]
            
            self._move_gripper_to(
                hover,
                finger_angle=open_finger,
                steps=150
                )

            self._settle_simulation(
                 40,
                 finger_angle=open_finger
                 )

            self._log_task_state("APPROACH", uid=uid, gripper_command="open", grasp_success=False, place_success=False)
           #self._move_gripper_to(approach, finger_angle=open_finger, steps=90)
            #self._settle_simulation(20, finger_angle=open_finger)
            #safe_hover = [
              #  object_center[0],
               # object_center[1],
                #0.55
              #  ]

            #self._move_gripper_to(
              #steps=100
               # )

            #self._settle_simulation(20)

            print('  Descending to grasp...')
            self._log_task_state("PRE_GRASP", uid=uid, gripper_command="open", grasp_success=False, place_success=False)
            self._move_gripper_to(pre_grasp, finger_angle=open_finger, steps=150)
            self._settle_simulation(20, finger_angle=open_finger)
            state = p.getLinkState(
                self._kuka.kukaUid,
                self._kuka.kukaEndEffectorIndex
                )
            print("PREGRASP TARGET =", pre_grasp)
            print("PREGRASP ACTUAL =", state[0])
            print("PREGRASP XY ERROR =",abs(state[0][0] - pre_grasp[0]),abs(state[0][1] - pre_grasp[1]))
            #error = self._gripper_xy_error(object_center[:2])

            #print("XY error =", error)

            #if error > 0.03:
             #  print("Not centered over object. Retrying.")
              # continue
           
            grasp_pos = self._move_gripper_to(lower, finger_angle=open_finger, steps=120)
            grasp_pos=lower
            self._settle_simulation(20, finger_angle=open_finger)

            if not self._object_between_fingers(uid):
                print("retry triggered")
               # safe_home = [
                #    self._picking_tray_center[0],
                 #   self._picking_tray_center[1],
                  #  0.60
                   # ]

            #if False:
                self._log_task_state("PRE_GRASP", uid=uid, gripper_command="open",
                                     grasp_success=False, place_success=False,
                                     reason="object_not_centered_between_fingers")
                    # move only slightly upward
                self._move_gripper_to(
                    pre_grasp,
                    finger_angle=open_finger,
                    steps=40
                    )

               # self._move_gripper_to(safe_home, finger_angle=open_finger, steps=120)
                self._settle_simulation(15,finger_angle=open_finger)
                continue

            print('  Closing gripper...')
            self._log_task_state("GRASP", uid=uid, gripper_command="closing", grasp_success=False, place_success=False)
            self._move_gripper_to(grasp_pos, finger_angle=0.15, steps=80)
            self._settle_simulation(25, finger_angle=0.15)
            self._move_gripper_to(grasp_pos, finger_angle=closed_finger, steps=85)
            self._settle_simulation(80, finger_angle=closed_finger)

            grasp_success = self._grasp_confirmed(uid, closed_finger_angle=closed_finger)
            self._log_task_state("GRASP", uid=uid, gripper_command="closed",
                                 grasp_success=grasp_success, place_success=False)
            if not grasp_success:
                self._log_task_state("GRASP", uid=uid, gripper_command="open",
                                     grasp_success=False, place_success=False,
                                     reason="invalid_grasp_contacts_reopen_for_retry")
                self._move_gripper_to(
                    pre_grasp,
                    finger_angle=open_finger,
                    steps=45
                    )
                self._settle_simulation(
                    80,
                    finger_angle=open_finger
                    )
                self._settle_simulation(30, finger_angle=open_finger)
                continue

            print('  Lifting object...')
            self._log_task_state("LIFT", uid=uid, gripper_command="closed", grasp_success=True, place_success=False)
            lift_pose = [object_center[0], object_center[1], 0.55]#object_top + 0.45]
            before_lift_z = p.getBasePositionAndOrientation(uid)[0][2]
            self._move_gripper_to(lift_pose, finger_angle=closed_finger, steps=180)
            self._settle_simulation(80, finger_angle=closed_finger)
            after_lift_z = p.getBasePositionAndOrientation(uid)[0][2]
            if after_lift_z <= before_lift_z + 0.03:
                self._log_task_state("LIFT", uid=uid, gripper_command="open",
                                     grasp_success=False, place_success=False,
                                     reason=f"lift_failed_before_transfer dz={after_lift_z - before_lift_z:.4f}")
                self._move_gripper_to(lift_pose, finger_angle=open_finger, steps=40)
                self._settle_simulation(40, finger_angle=open_finger)
                continue
            self._log_task_state("LIFT", uid=uid, gripper_command="closed", grasp_success=True, place_success=False,
                                 reason=f"lift_verified dz={after_lift_z - before_lift_z:.4f}")
            return True

        return False

    def place_object(self, uid):
        open_finger = 0.30
        closed_finger = 0.0
        place_pos, _ = p.getBasePositionAndOrientation(self._placement_tray_uid)
        place_center = [place_pos[0], place_pos[1], place_pos[2] + 0.02]
        place_approach = [place_center[0], place_center[1], 0.60]#place_center[2] + 0.45]
        place_lower = [place_center[0], place_center[1], place_center[2] + 0.02]#place_center[2] + 0.02]

        # Move to placement tray approach.
        print('  Moving to placement tray...')
        self._log_task_state("TRANSFER", uid=uid, gripper_command="closed", grasp_success=True, place_success=False)
        obj_pos, _ = p.getBasePositionAndOrientation(uid)
        transfer_mid = [obj_pos[0], place_center[1], place_center[2] + 0.55]
        self._move_gripper_to(transfer_mid, finger_angle=closed_finger, steps=80)
        self._move_gripper_to(place_approach, finger_angle=closed_finger, steps=120)
        self._settle_simulation(50, finger_angle=closed_finger)
        if not self._grasp_confirmed(uid, closed_finger_angle=closed_finger):
            self._log_task_state("TRANSFER", uid=uid, gripper_command="closed",
                                 grasp_success=False, place_success=False,
                                 reason="grasp_lost_before_pre_place")
            raise RuntimeError(f'Object {uid} slipped before destination tray')
        #safe_place_hover = [
            place_center[0],
            place_center[1],
            0.55
         #   ]

        #self._move_gripper_to(
            safe_place_hover,
            finger_angle=closed_finger,
            steps=100
         #   )
        #self._settle_simulation(20)
        print('  Descending into placement tray...')
        self._log_task_state("PRE_PLACE", uid=uid, gripper_command="closed", grasp_success=True, place_success=False)
        self._move_gripper_to(place_lower, finger_angle=closed_finger, steps=85)
        self._settle_simulation(25, finger_angle=closed_finger)

        #if not self._gripper_over_placing_tray():
         #   self._log_task_state("PRE_PLACE", uid=uid, gripper_command="closed",
          #                       grasp_success=True, place_success=False,
           #                      reason="not_over_destination_tray")
            #raise RuntimeError('Refusing to open gripper outside destination tray')
        #print("Skipping destination tray safety check")
        if not self._gripper_over_placing_tray():

           print("Not centered over destination tray. Repositioning.")
           print("proceeding with thw placement anyways.")

           #place_approach = [
            #   place_center[0],
             #  place_center[1],
              # 0.60
               #]

           self._move_end_effector_to(
               place_approach,
               finger_angle=closed_finger,
               steps=120
               )

           self._settle_simulation(
               50,
               finger_angle=closed_finger
            )
           if not self._gripper_over_placing_tray():
               print("proceeding with thw placement anyways.")
          # raise RuntimeError(#continue
           #     "Not centered over destination tray"
            #    )
        state = p.getLinkState(
            self._kuka.kukaUid,
            self._kuka.kukaEndEffectorIndex
            )

        print("PLACE POSITION =", state[0])
        print("TARGET PLACE =", place_lower)
        print('  Releasing object...')
        self._log_task_state(
             "PLACE",
             uid=uid,
             gripper_command="closed",
             grasp_success=True,
             place_success=False
             )

         # descend while still gripping
        self._move_gripper_to(
            place_lower,
            finger_angle=closed_finger,
            steps=120
            )

        # let arm settle
        self._settle_simulation(
            50,
            finger_angle=closed_finger
            )
        # now open
        self._move_gripper_to(
            place_lower,
            finger_angle=open_finger,
            steps=30
            )
        # allow object to settle in tray
        self._settle_simulation(
            250,
            finger_angle=open_finger
            )
        pos, _ = p.getBasePositionAndOrientation(uid)
        if not self._is_object_in_placing_tray(pos):
            self._log_task_state("PLACE", uid=uid, gripper_command="open",
                                 grasp_success=False, place_success=False,
                                 reason="released_outside_destination_tray")
            raise RuntimeError(f'Object {uid} was released outside destination tray')

        #print('  Retracting...')
        #self._log_task_state("RETRACT", uid=uid, gripper_command="open", grasp_success=False, place_success=True)
        #self._move_gripper_to(place_approach, finger_angle=open_finger, steps=75)
        #self._settle_simulation(25)
        #return True
        print('  Retracting...')

        self._move_gripper_to(
         place_approach,
         finger_angle=open_finger,
         steps=75
         )

        source_hover = [
         self._picking_tray_center[0],
         self._picking_tray_center[1],
         0.55
         ]

        print('  Returning to source tray...')

        self._move_gripper_to(
             source_hover,
             finger_angle=open_finger,
             steps=180
             )

        self._settle_simulation(60)
        return True

    def _move_gripper_to(self, position, finger_angle=0.0, steps=80):
     orn = p.getQuaternionFromEuler([0,-np.pi, 0])#3.14159

     state = p.getLinkState(
         self._kuka.kukaUid,
         self._kuka.kukaEndEffectorIndex
     )

     start = np.array(state[0])
     target = np.array(position)

     for i in range(steps):
         t=float(i+1)/float(steps)
         #alpha = float(i + 1) / float(steps)
         alpha = 3*t*t-2*t*t*t

         current_target = (
             start * (1.0 - alpha)
             + target * alpha
         )

         joint_poses = p.calculateInverseKinematics(
             self._kuka.kukaUid,
             self._kuka.kukaEndEffectorIndex,
             current_target.tolist(),
             orn,
             self._kuka.ll,
             self._kuka.ul,
             self._kuka.jr,
             self._kuka.rp,
         )

         for joint_index in range(self._kuka.kukaEndEffectorIndex + 1):

             p.setJointMotorControl2(
                 bodyUniqueId=self._kuka.kukaUid,
                 jointIndex=joint_index,
                 controlMode=p.POSITION_CONTROL,
                 targetPosition=joint_poses[joint_index],
                 targetVelocity=0,
                 force=self._kuka.maxForce,
                 maxVelocity=self._kuka.maxVelocity*0.4,
                 positionGain=0.8,
                 velocityGain=1,
             )

         self._control_fingers(finger_angle)

         p.stepSimulation()

         if self._renders:
             time.sleep(self._timeStep)

    def _descend_to_grasp_height(self, target_position, finger_angle=0.30, max_steps=80):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        print(
            "DESCEND POS:",
            state[0]
            )
        current = list(state[0])
        target = list(target_position)
        for _ in range(max_steps):
            dz = max(target[2] - current[2], -0.005)
            if abs(dz) < 1e-4:
                return current
            current = [current[0], current[1], current[2] + dz]
            self._move_gripper_to(current, finger_angle=finger_angle, steps=1)
        return current

    def _finger_contact_sides(self, uid):
        contacts = p.getContactPoints(bodyA=self._kuka.kukaUid, bodyB=uid)
        left_links = {8, 10}
        right_links = {11, 13}
        touched_left = any(contact[3] in left_links for contact in contacts)
        touched_right = any(contact[3] in right_links for contact in contacts)
        return touched_left, touched_right

    def _object_between_fingers(self, uid):
        obj_pos, _ = p.getBasePositionAndOrientation(uid)
        left_pos = p.getLinkState(self._kuka.kukaUid, 10)[0]
        right_pos = p.getLinkState(self._kuka.kukaUid, 13)[0]
        midpoint = np.array(left_pos) * 0.5 + np.array(right_pos) * 0.5
        finger_gap = np.linalg.norm(np.array(left_pos) - np.array(right_pos))
        lateral_error = np.linalg.norm(np.array(obj_pos[:2]) - midpoint[:2])
        vertical_error = abs(obj_pos[2] - midpoint[2])
        return lateral_error < max(0.055, finger_gap * 0.75) and vertical_error < 0.08

    def _gripper_closed_enough(self, closed_finger_angle):
        joint_8 = abs(p.getJointState(self._kuka.kukaUid, 8)[0])
        joint_11 = abs(p.getJointState(self._kuka.kukaUid, 11)[0])
        return joint_8 <= closed_finger_angle + 0.08 and joint_11 <= closed_finger_angle + 0.08

    def _grasp_confirmed(self, uid, closed_finger_angle=0.0):
        touched_left, touched_right = self._finger_contact_sides(uid)
        return (#self._gripper_closed_enough(closed_finger_angle) and
                self._object_between_fingers(uid) and
                (touched_left or touched_right))

    def _gripper_over_placing_tray(self):
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        gripper_pos = state[0]
        tray_pos, _ = p.getBasePositionAndOrientation(self._placement_tray_uid)
        return abs(gripper_pos[0] - tray_pos[0]) < 0.18 and abs(gripper_pos[1] - tray_pos[1]) < 0.16

    def _control_fingers(self, finger_angle):
        p.setJointMotorControl2(self._kuka.kukaUid, 7, p.POSITION_CONTROL,
                                targetPosition=self._kuka.endEffectorAngle,
                                force=self._kuka.maxForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL,
                                targetPosition=-finger_angle,
                                force=80)
        p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL,
                                targetPosition=finger_angle,
                                force=80)
        p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self._kuka.fingerTipForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self._kuka.fingerTipForce)

    def _move_end_effector_to(self, position, finger_angle=0.0, steps=140):
        for _ in range(steps):
            state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
            current = state[0]
            delta = [position[0] - current[0], position[1] - current[1], position[2] - current[2]]
            action = [
                np.clip(delta[0], -0.012, 0.012),
                np.clip(delta[1], -0.012, 0.012),
                np.clip(delta[2], -0.012, 0.012),
                0.0,
                finger_angle,
            ]
            self._kuka.applyAction(action)
            for _ in range(4):
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)

    def _settle_simulation(self, steps=80, finger_angle=None):
        for _ in range(steps):
            if finger_angle is not None:
                self._control_fingers(finger_angle)
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)


# Device configuration (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def make_env(renders=True):
    return CustomKukaEnv(renders=renders,
                         cameraRandom=0.75,
                         isDiscrete=False,
                         removeHeightHack=False,
                         maxSteps=10,
                         numObjects=5)


# Existing training scripts import a module-level env from here.
# Standalone visual scripts can opt out so they own the only physics client.
env = None if os.environ.get("KUKA_SKIP_GLOBAL_ENV") == "1" else make_env(renders=False)
