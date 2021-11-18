import json
import time
from collections import OrderedDict
from typing import Generator, cast
from dataclasses import dataclass
from pathlib import Path
import torch

import gym
import gym.spaces as spaces
import gym.utils
import gym.utils.seeding
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import GPT2Tokenizer

GUI = False
VIEW_MATRIX = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0],
    distance=2,
    yaw=0,
    pitch=10,
    roll=0,
    upAxisIndex=2,
)

PROJECTION_MATRIX = p.computeProjectionMatrixFOV(
    fov=50, aspect=1, nearVal=0.01, farVal=10
)


@dataclass
class PointMassEnv(gym.GoalEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}
    sparse_rew_thresh: float
    mission_nvec: np.ndarray
    tokenizer: GPT2Tokenizer
    image_width: float = 96
    image_height: float = 72
    env_bounds: float = 2.5
    cameraYaw: float = 35

    def __post_init__(
        self,
    ):

        self._max_episode_steps = 100
        self.action_space = spaces.Discrete(5)

        def urdfs():
            for subdir in Path("dataset").iterdir():
                urdf = Path(subdir, "mobility.urdf")
                assert urdf.exists()
                with Path(subdir, "meta.json").open() as f:
                    meta = json.load(f)
                yield meta["model_cat"], urdf

        self.urdfs = OrderedDict(urdfs())

        def tokens() -> Generator[torch.Tensor, None, None]:
            for k in self.urdfs:
                encoded = self.tokenizer.encode(k, return_tensors="pt")
                tensor = cast(torch.Tensor, encoded)
                yield tensor.squeeze(0)

        padded = pad_sequence(
            list(tokens()),
            padding_value=self.tokenizer.eos_token_id,
        ).T

        self.tokens = OrderedDict(zip(self.urdfs, padded))

        self.observation_space = spaces.Dict(
            dict(
                mission=spaces.MultiDiscrete(
                    np.ones_like(self.mission_nvec[0]) * padded.max().item()
                ),
                image=spaces.Box(
                    low=0,
                    high=255,
                    shape=[self.image_width, self.image_height, 3],
                ),
            )
        )

        self.is_render = False
        self.gui_active = False
        self._p = p
        self.physics_client_active = 0
        self._seed()
        self.iterator = None

        self.relativeChildPosition = [0, 0, 0]
        self.relativeChildOrientation = [0, 0, 0, 1]

    def get_observation(self):
        (_, _, rgbPixels, _, _,) = p.getCameraImage(
            self.image_width,
            self.image_height,
            viewMatrix=VIEW_MATRIX,
            projectionMatrix=PROJECTION_MATRIX,
            shadow=0,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        return dict(
            image=rgbPixels.astype(np.float32),
            mission=self.tokens[self.mission],
        )

    def compute_reward_sparse(self, achieved_goal, desired_goal):

        initially_vectorized = True
        dimension = 2
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.expand_dims(np.array(achieved_goal), axis=0)
            desired_goal = np.expand_dims(np.array(desired_goal), axis=0)
            initially_vectorized = False

        reward = np.zeros(len(achieved_goal))
        for g in range(0, len(achieved_goal[0]) // dimension):  # piecewise reward
            g = g * dimension  # increments of 2
            current_distance = np.linalg.norm(
                achieved_goal[:, g : g + dimension]
                - desired_goal[:, g : g + dimension],
                axis=1,
            )
            reward += np.where(current_distance > self.sparse_rew_thresh, -1, 0)

        if not initially_vectorized:
            return reward[0]
        else:
            return reward

    def generator(self):
        missions = []
        goals = []

        for base_position in [
            [self.env_bounds, self.env_bounds, 0],
            [-self.env_bounds, -self.env_bounds, 0],
        ]:
            urdfs = list(self.urdfs.items())
            name, urdf = urdfs[self.np_random.choice(len(urdfs))]
            missions.append(name)

            try:
                goal = self._p.loadURDF(
                    str(urdf), basePosition=base_position, useFixedBase=True
                )
            except p.error:
                print(p.error)
                raise RuntimeError(f"Error while loading {urdf}")
            goals.append(goal)

            collisionFilterGroup = 0
            collisionFilterMask = 0
            self._p.setCollisionFilterGroupMask(
                goal, -1, collisionFilterGroup, collisionFilterMask
            )
            self._p.createConstraint(
                goal,
                -1,
                -1,
                -1,
                self._p.JOINT_FIXED,
                [1, 1, 1.4],
                [0, 0, 0],
                self.relativeChildPosition,
                self.relativeChildOrientation,
            )
        self.mission = self.np_random.choice(missions)

        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, GUI)

        self._p.setGravity(0, 0, -10)
        look_at = [0, 0, 0.1]
        distance = 7
        yaw = 0
        self._p.resetDebugVisualizerCamera(distance, yaw, -89, look_at)
        cybe = self._p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1])
        visplaneId = self._p.createVisualShape(
            p.GEOM_BOX, halfExtents=[5, 5, 0.1], rgbaColor=[1, 1, 1, 1]
        )
        self._p.createMultiBody(0, cybe, visplaneId, [0, 0, -0.2])

        self._p.resetBasePositionAndOrientation(self.mass, [0, 0, 0.6], [0, 0, 0, 1])
        action = yield self.get_observation()

        for global_step in range(self._max_episode_steps):
            self.apply_action(action)
            s = self.get_observation()
            r = 0
            t = False
            action = yield s, r, t, {}

        self.apply_action(action)
        s = self.get_observation()
        r = 0
        t = True
        for goal in goals:
            self._p.removeBody(goal)
        yield s, r, t, {}

    def apply_action(self, action):
        x_shift, y_shift, *_ = action * 0.1  # put it to the correct scale
        x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
        new_x = np.clip(x + x_shift, -2 * self.env_bounds, 2 * self.env_bounds)
        new_y = np.clip(y + y_shift, -2 * self.env_bounds, 2 * self.env_bounds)
        self._p.changeConstraint(self.mass_cid, [new_x, new_y, -0.1], maxForce=10)
        for i in range(0, 20):
            self._p.stepSimulation()

    def step(self, action: np.ndarray):
        return self.iterator.send(action)

    def reset(self):
        if not self.physics_client_active:
            if self.is_render:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
                self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
                self.gui_active = True
            else:
                self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

            self.physics_client_active = 1

            sphereRadius = 0.2
            mass = 1
            visualShapeId = 2
            colSphereId = self._p.createCollisionShape(
                p.GEOM_SPHERE, radius=sphereRadius
            )
            self.mass = self._p.createMultiBody(
                mass, colSphereId, visualShapeId, [0, 0, 0.4]
            )

            self.mass_cid = self._p.createConstraint(
                self.mass,
                -1,
                -1,
                -1,
                self._p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0],
                self.relativeChildPosition,
                self.relativeChildOrientation,
            )

        self.iterator = self.generator()
        return next(self.iterator)

    def render(self, mode="human"):

        if mode == "human":
            self.is_render = True
            return np.array([])
        if mode == "rgb_array":
            raise NotImplementedError

    def close(self):
        print("closing")
        self._p.disconnect()

    def _seed(self, seed=None):
        print("seeding")
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    env = PointMassEnv(
        sparse_rew_thresh=0.3,
        mission_nvec=np.array([400] * 8),
        image_width=84,
        image_height=84,
        tokenizer=tokenizer,
    )
    env.render(mode="human")
    env.reset()

    forward = 0
    turn = 0

    cameraDistance = 3
    cameraYaw = 35
    cameraPitch = -45
    steps = 0
    while True:
        try:

            spherePos, orn = p.getBasePositionAndOrientation(env.mass)

            cameraTargetPosition = spherePos
            p.resetDebugVisualizerCamera(
                cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition
            )
            camInfo = p.getDebugVisualizerCamera()
            camForward = camInfo[5]

            keys = p.getKeyboardEvents()
            for k, v in keys.items():

                right = k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)
                right_release = k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)
                left = k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)
                left_release = k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)
                up = k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)
                up_release = k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)
                down = k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)
                down_release = k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)
                no_op = right_release or left_release or up_release or down_release

                action = [
                    right,
                    right_release,
                    left,
                    left_release,
                    up,
                    up_release,
                    down,
                    down_release,
                ]

                if right:
                    turn = -3
                if right_release:
                    turn = 0
                if left:
                    turn = 3
                if left_release:
                    turn = 0

                if up:
                    forward = 1.8
                if up_release:
                    forward = 0
                if down:
                    forward = -1.8
                if down_release:
                    forward = 0

            force = [forward * camForward[0], forward * camForward[1], 0]
            cameraYaw = cameraYaw + turn

            time.sleep(3.0 / 240.0)

            o, r, t, _ = env.step(np.array(force))
            if t:
                env.reset()
            # print(o["observation"][0:2])  # print(r)

            steps += 1
        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting.")
            return


if __name__ == "__main__":
    main()
