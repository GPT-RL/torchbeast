import gym
import numpy as np
import torch
from gym import spaces
from torch import nn

from torchbeast import atari_wrappers, monobeast
from torchbeast.core import environment
from torchbeast.core.environment import _format_frame
from torchbeast.environment import Observation, ObservationSpace, PointMassEnv
from torchbeast.lazy_frames import LazyFrames
from torchbeast.monobeast import Args


class ImageToPyTorch(atari_wrappers.ImageToPyTorch):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_spaces = ObservationSpace(*self.observation_space.spaces)
        old_shape = obs_spaces.image.shape
        self.observation_space = gym.spaces.Tuple(
            ObservationSpace(
                mission=obs_spaces.mission,
                image=gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(old_shape[-1], old_shape[0], old_shape[1]),
                    dtype=np.uint8,
                ),
            )
        )

    def observation(self, observation):
        obs = Observation(*observation)
        image = super().observation(obs.image)
        obs = Observation(image=image, mission=obs.mission)
        # assert self.observation_space.contains(obs)
        return obs


class Environment(environment.Environment):
    def _initial(self, obs):
        obs = Observation(*obs)
        initial = super()._initial(obs.image)
        return dict(**initial, mission=torch.tensor(obs.mission))

    def _step(self, frame, action, reward, done):
        obs = Observation(*frame)
        frame = obs.image
        mission = obs.mission
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            obs = Observation(*frame)
            frame = obs.image
            mission = obs.mission
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        return dict(
            mission=torch.tensor(mission),
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )


class Network(monobeast.AtariNet):
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
        mission = inputs["mission"]
        mission = self.mission_encoder.forward(mission.reshape(-1, mission.size(-1)))
        return T, B, torch.cat([core_input, mission], dim=-1)

    @staticmethod
    def get_fc():
        return nn.Linear(2560, 512)


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
        obs = Observation(image=super().observation(obs.image), mission=obs.mission)
        # assert self.observation_space.contains(obs)
        return obs


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
        obs = Observation(image=image, mission=mission)
        # assert self.observation_space.contains(obs)
        return obs


class Trainer(monobeast.Trainer):
    @staticmethod
    def wrap_env(gym_env):
        return Environment(gym_env)

    @staticmethod
    def create_env(args: Args):
        env = PointMassEnv(reindex_tokens=True)
        env = ScaledFloatFrame(env)
        env = FrameStack(env, 4)
        env = ImageToPyTorch(env)
        return env

    @classmethod
    def buffer_specs(cls, obs_space: spaces.Tuple, num_actions, T):
        space = ObservationSpace(*obs_space.spaces)
        specs = super().buffer_specs(space.image, num_actions, T)
        nvec = space.mission.nvec
        return dict(mission=dict(size=[T + 1, *nvec.shape], dtype=torch.int32), **specs)

    @classmethod
    def build_net(cls, args, gym_env):
        return Network(
            observation_space=ObservationSpace(*gym_env.observation_space.spaces),
            num_actions=gym_env.action_space.n,
            use_lstm=args.use_lstm,
        )


if __name__ == "__main__":
    Trainer().main(Args().parse_args())
