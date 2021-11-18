from dataclasses import dataclass
from pathlib import Path

import gym, gym.utils, gym.utils.seeding
import pybullet as p
import numpy as np
import time
from pybullet_utils import bullet_client

import gym.spaces as spaces
import math

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
    env_bounds: float = 2.5
    num_goals: int = 2

    def __post_init__(
        self,
    ):

        action_dim = 2
        obs_dim = 4

        self._max_episode_steps = 100

        self.num_goals = 1
        goal_dim = 2 * self.num_goals

        high = np.ones([action_dim])
        self.action_space = spaces.Box(-high, high)
        high_obs = self.env_bounds * np.ones([obs_dim])
        high_goal = self.env_bounds * np.ones([goal_dim])
        self.urdfs = [str(u) for u in Path("dataset").glob("*/mobility.urdf")]

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(-high_goal, high_goal),
                achieved_goal=spaces.Box(-high_goal, high_goal),
                observation=spaces.Box(-high_obs, high_obs),
                controllable_achieved_goal=spaces.Box(
                    -self.env_bounds * np.ones([action_dim]),
                    self.env_bounds * np.ones([action_dim]),
                ),
                full_positional_state=spaces.Box(
                    -self.env_bounds * np.ones([action_dim]),
                    self.env_bounds * np.ones([action_dim]),
                ),
            )
        )

        self.is_render = False
        self.gui_active = False
        self._p = p
        self.physics_client_active = 0
        self.target_min = 0.1
        self._seed()
        self.global_step = 0
        self.opposite_goal = False
        self.objects = []

    @staticmethod
    def crop(num, lim):
        if 0 <= num < lim:
            num = lim
        elif 0 > num > -lim:
            num = -lim
        return num

    def initialize_actor_pos(self, o):
        x, y, x_vel, y_vel = o[0], o[1], o[2], o[3]
        self._p.resetBasePositionAndOrientation(self.mass, [x, y, -0.1], [0, 0, 0, 1])
        self._p.changeConstraint(self.mass_cid, [x, y, -0.1], maxForce=100)
        self._p.resetBaseVelocity(self.mass, [x_vel, y_vel, 0])

    # TODO change the env initialise start pos to a more general form of the function

    def initialize_start_pos(self, o):
        if isinstance(o, dict):
            o = o["observation"]
        self.initialize_actor_pos(o)

    def calc_state(self):

        # state will be x,y pos, total velocity, x,y goal.
        current_pos = self._p.getBasePositionAndOrientation(self.mass)[0]
        x, y = current_pos[0], current_pos[1]
        velocity = self._p.getBaseVelocity(self.mass)[0]
        x_vel, y_vel = velocity[0], velocity[1]
        obs = [x, y, x_vel, y_vel]

        achieved_goal = np.array([x, y])
        extra_info = None

        full_positional_state = np.array([x, y])
        return_dict = dict(
            observation=np.array(obs).copy().astype("float32"),
            achieved_goal=np.array(achieved_goal).copy().astype("float32"),
            desired_goal=np.zeros(2).astype("float32"),
            extra_info=extra_info,
            controllable_achieved_goal=np.array([x, y]).copy().astype("float32"),
            full_positional_state=full_positional_state.astype("float32"),
        )

        if self.is_render:
            img = p.getCameraImage(
                width=48,
                height=48,
                viewMatrix=VIEW_MATRIX,
                projectionMatrix=PROJECTION_MATRIX,
                shadow=0,
                flags=p.ER_NO_SEGMENTATION_MASK,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            return_dict["image"] = img[2][:, :, :3]

        return return_dict

    @staticmethod
    def calc_target_distance(achieved_goal, desired_goal):
        distance = np.sum(abs(achieved_goal - desired_goal))
        return distance

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info=None
    ):
        # reward given if new pos is closer than old
        current_distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        position_reward = -1000 * (current_distance - self.last_target_distance)
        self.last_target_distance = current_distance
        return position_reward

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

    def step(self, action: np.ndarray):

        x_shift, y_shift, *_ = action * 0.1  # put it to the correct scale
        x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]

        new_x = np.clip(x + x_shift, -2 * self.env_bounds, 2 * self.env_bounds)
        new_y = np.clip(y + y_shift, -2 * self.env_bounds, 2 * self.env_bounds)
        self._p.changeConstraint(self.mass_cid, [new_x, new_y, -0.1], maxForce=10)

        for i in range(0, 20):
            self._p.stepSimulation()

        obs = self.calc_state()

        r = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])

        self.global_step += 1

        success = (
            0
            if self.compute_reward_sparse(obs["achieved_goal"], obs["desired_goal"]) < 0
            else 1
        )  # assuming negative rewards

        # this part is only for interoperability with baselines
        if self.global_step == self._max_episode_steps:
            done = True
        else:
            done = False
        return obs, r, done, {"is_success": success}

    def reset(self, o=None):
        if o is not None:
            self.initialize_start_pos(o)
        else:
            self.global_step = 0

            if self.physics_client_active == 0:

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
                relativeChildPosition = [0, 0, 0]
                relativeChildOrientation = [0, 0, 0, 1]
                self.mass_cid = self._p.createConstraint(
                    self.mass,
                    -1,
                    -1,
                    -1,
                    self._p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    relativeChildPosition,
                    relativeChildOrientation,
                )

                self.goal_cids = []

                for g in range(0, self.num_goals):

                    urdf = self.np_random.choice(self.urdfs)

                    goal = p.loadURDF(
                        urdf, basePosition=[0.5, 0.0, 0.5], useFixedBase=True
                    )

                    collisionFilterGroup = 0
                    collisionFilterMask = 0
                    self._p.setCollisionFilterGroupMask(
                        goal, -1, collisionFilterGroup, collisionFilterMask
                    )
                    self.goal_cids.append(
                        self._p.createConstraint(
                            goal,
                            -1,
                            -1,
                            -1,
                            self._p.JOINT_FIXED,
                            [1, 1, 1.4],
                            [0, 0, 0],
                            relativeChildPosition,
                            relativeChildOrientation,
                        )
                    )

                if GUI:
                    ACTION_LIMIT = 1
                    self.x_shift = self._p.addUserDebugParameter(
                        "X", -ACTION_LIMIT, ACTION_LIMIT, 0.0
                    )
                    self.y_shift = self._p.addUserDebugParameter(
                        "Y", -ACTION_LIMIT, ACTION_LIMIT, 0.0
                    )

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

            self._p.resetBasePositionAndOrientation(
                self.mass, [0, 0, 0.6], [0, 0, 0, 1]
            )

            # reset mass location
            x = self.crop(
                self.np_random.uniform(low=-self.env_bounds, high=self.env_bounds),
                self.target_min,
            )
            y = self.crop(
                self.np_random.uniform(low=-self.env_bounds, high=self.env_bounds),
                self.target_min,
            )

            x_vel = 0  # self.np_random.uniform(low=-1, high=1)
            y_vel = 0  # self.np_random.uniform(low=-1, high=1)

            self.initialize_actor_pos([x, y, x_vel, y_vel])

        obs = self.calc_state()
        self.last_target_distance = self.calc_target_distance(
            obs["achieved_goal"], obs["desired_goal"]
        )

        return obs

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
    env = PointMassEnv(sparse_rew_thresh=0.3)
    env.render(mode="human")
    env.reset()["observation"]

    forward = 0
    turn = 0

    cameraDistance = 3
    cameraYaw = 35
    cameraPitch = -45
    steps = 0
    while steps < 3000:

        spherePos, orn = p.getBasePositionAndOrientation(env.mass)

        cameraTargetPosition = spherePos
        p.resetDebugVisualizerCamera(
            cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition
        )
        camInfo = p.getDebugVisualizerCamera()
        camForward = camInfo[5]

        keys = p.getKeyboardEvents()
        for k, v in keys.items():

            if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                turn = -3
            if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
                turn = 0
            if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                turn = 3
            if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
                turn = 0

            if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
                forward = 1.8
            if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
                forward = 0
            if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
                forward = -1.8
            if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
                forward = 0

        force = [forward * camForward[0], forward * camForward[1], 0]
        cameraYaw = cameraYaw + turn

        time.sleep(3.0 / 240.0)

        o, r, _, _ = env.step(np.array(force))
        print(o["observation"][0:2])  # print(r)

        steps += 1


if __name__ == "__main__":
    main()
