from colosseumrl import BaseEnvironment
from gym.spaces import Space
from ray.rllib import MultiAgentEnv

from typing import Dict


class RllibWrapper(MultiAgentEnv):
    # Overwrite the following methods
    def create_env(self, *args, **kwargs) -> BaseEnvironment:
        raise NotImplementedError

    def create_observation_space(self, *args, **kwargs) -> Space:
        raise NotImplementedError

    def create_action_space(self, *args, **kwargs) -> Space:
        raise NotImplementedError

    def create_done_dict(self, state, players, rewards, terminal, action_dict) -> Dict[str, bool]:
        return {player: terminal for player in action_dict}

    def create_info_dict(self, state, players, rewards, terminal, action_dict) -> Dict:
        return {}

    def action_map(self, action):
        return action

    # You can keep these the same most of the time
    def __init__(self, *args, **kwargs):
        self.env = self.create_env(*args, **kwargs)
        self.action_space = self.create_action_space(*args, **kwargs)
        self.observation_space = self.create_observation_space(*args, **kwargs)

        self.state = None
        self.players = None

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in self.players}

    def step(self, action_dict):
        actions = [self.action_map(action_dict[player]) if player in action_dict else ''
                   for player in map(str, self.players)]

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        observations = {player: self.env.state_to_observation(self.state, int(player)) for player in action_dict}
        reward_dict = {player: rewards[int(player)] for player in action_dict}

        done_dict = self.create_done_dict(self.state, self.players, rewards, winners, action_dict)
        done_dict['__all__'] = terminal

        info_dict = self.create_info_dict(self.state, self.players, rewards, winners, action_dict)

        return observations, reward_dict, done_dict, info_dict
