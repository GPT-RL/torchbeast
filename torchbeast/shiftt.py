from collections import deque

import gym
import numpy as np
import torch
from gym import spaces
from torch import nn

from torchbeast import atari_wrappers, monobeast, net
from torchbeast.util import LazyFrames
from torchbeast.core import environment
from torchbeast.core.environment import _format_frame
from torchbeast.environment import Observation, ObservationSpace, PointMassEnv
from torchbeast.monobeast import BufferSpec, parser


class Environment(environment.Environment):
    @staticmethod
    def _format_frame(frame):
        frame = torch.from_numpy(frame)
        return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).

    def _initial(self, obs):
        obs = Observation(*obs)
        initial = super()._initial(obs.image)
        return dict(**initial, mission=_format_frame(obs.mission))

    def _step(self, frame, action, reward, done):
        obs = Observation(*frame)
        step = super()._step(obs.image, action, reward, done)
        return dict(**step, mission=_format_frame(obs.mission))


class Network(net.AtariNet):
    def __init__(
        self, observation_space: ObservationSpace, num_actions: int, use_lstm: bool
    ):
        nvec = observation_space.mission.nvec
        self.embedding_dim = 64
        super().__init__(observation_space.image.shape, num_actions, use_lstm)
        self.mission_encoder = nn.EmbeddingBag(nvec.max().item(), self.embedding_dim)

    def get_core_output_size(self, num_actions):
        return super().get_core_output_size(num_actions) + self.embedding_dim

    def get_core_input(self, inputs):
        T, B, core_input = super().get_core_input(inputs)
        mission = self.mission_encoder(inputs["mission"])
        return T, B, torch.cat([core_input, mission], dim=-1)


class ScaledFloatFrame(atari_wrappers.ScaledFloatFrame):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_spaces = ObservationSpace(*self.observation_space.spaces)
        self.observation_space = gym.spaces.Tuple(
            ObservationSpace(
                mission=obs_spaces.mission,
                image=gym.spaces.Box(
                    low=0, high=1, shape=obs_spaces.image.shape, dtype=np.float32
                ),
            )
        )

    def observation(self, obs):
        obs = Observation(*obs)
        return Observation(image=super().observation(obs.image), mission=obs.mission)


class FrameStack(atari_wrappers.FrameStack):
    def __init__(self, env, k):
        obs_spaces = ObservationSpace(*env.observation_space.spaces)
        super().__init__(env, k)
        self.observation_space = spaces.Tuple(
            ObservationSpace(mission=obs_spaces.mission, image=self.observation_space)
        )

    @staticmethod
    def get_observation_space(env):
        return ObservationSpace(*env.observation_space.spaces).image

    def _get_ob(self):
        assert len(self.frames) == self.k
        image = LazyFrames([Observation(*f).image for f in self.frames])
        mission = Observation(*self.frames[0]).mission
        return Observation(image=image, mission=mission)


class Trainer(monobeast.Trainer):
    @staticmethod
    def wrap_env(gym_env):
        return Environment(gym_env)

    @staticmethod
    def create_env(flags):
        env = PointMassEnv()
        env = ScaledFloatFrame(env)
        env = FrameStack(env, 4)
        return env

    @classmethod
    def buffer_specs(cls, obs_space: spaces.Tuple, num_actions, T):
        space = ObservationSpace(*obs_space.spaces)
        specs = super().buffer_specs(space.image, num_actions, T)
        nvec = space.mission.nvec
        return dict(mission=BufferSpec(size=nvec.shape, dtype=torch.uint8), **specs)

    @classmethod
    def build_network(cls, flags, gym_env):
        return Network(
            observation_space=ObservationSpace(*gym_env.observation_space.spaces),
            num_actions=gym_env.action_space.n,
            use_lstm=flags.use_lstm,
        )


if __name__ == "__main__":
    flags = parser.parse_args()
    Trainer().main(flags)
