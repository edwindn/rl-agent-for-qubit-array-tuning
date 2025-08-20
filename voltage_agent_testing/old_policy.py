from typing import Union, Type, Dict, Any, Optional, Tuple
import numpy as np
import torch as th
import torch
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import gymnasium as gym
from functools import partial
from tqdm import tqdm

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, PyTorchObs
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import make_proba_distribution, Distribution
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import zip_strict
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from model.voltage_agent import Agent

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, agent, device, feature_dim: int = None):
        if feature_dim is None:
            feature_dim = getattr(agent, 'feature_dim', 128)

        super().__init__(observation_space, feature_dim)
        self.agent = agent
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, dict):
            img = obs.get('image')
            voltages = obs.get('voltages')

            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            if not isinstance(voltages, torch.Tensor):
                voltages = torch.tensor(voltages)

            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[-1] in [1, 3]:
                img = img.permute(0, 3, 1, 2)
                
            if voltages.ndim == 1:
                voltages = voltages.unsqueeze(0)
            
            img = img.to(torch.float32).to(self.device)
            voltages = voltages.to(torch.float32).to(self.device)
            return self.agent.encode_obs(img, voltages)
        else:
            raise ValueError(f"Expected obs to be a dict, got {type(obs)}")



class CustomAgentPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        agent_class: Type = Agent,
        agent_kwargs: Dict[str, Any] = None,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        device: torch.device = None,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_extractor_class = CustomFeatureExtractor
        self.agent_kwargs = agent_kwargs or {}
        feature_extractor_kwargs = {"agent": agent_class(device=device, **self.agent_kwargs), "device": device}

        super().__init__(
            observation_space,
            action_space,
            feature_extractor_class,
            feature_extractor_kwargs,
            **kwargs
        )
        self.agent_class = agent_class
        self.agent = feature_extractor_kwargs["agent"]

        self.optimizer = self.optimizer_class(self.agent.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.feature_extractor = feature_extractor_class(
            observation_space=observation_space,
            agent=self.agent,
            device=device,
        )
        self.feature_dim = self.feature_extractor.feature_dim


    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                agent_class=self.agent_class,
                agent_kwargs=self.agent_kwargs,
                activation_fn=self.activation_fn,
                ortho_init=self.ortho_init,
                log_std_init=self.log_std_init,
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data


    def _process_recurrent_states(
        self,
        features: torch.Tensor,
        recurrent_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        update_recurrent: bool = True,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the recurrent states given the new observation
        and resets the recurrent states in the case of episodes initialising
        """

        if isinstance(recurrent_states, RNNStates):
            raise ValueError("Expected recurrent_states to be a tuple (h, c), got full RNNStates instead.")


        n_seq = recurrent_states[0].shape[1]
        feature_dim = features.shape[-1]
        features_sequence = features.reshape((n_seq, -1, feature_dim)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if torch.all(episode_starts == 0.0):
            recurrent_output, recurrent_states = self.agent.rssm(recurrent_states, features_sequence)
            recurrent_output = torch.flatten(recurrent_output.transpose(0, 1), start_dim=0, end_dim=1)
            return recurrent_output, recurrent_states


        # print('\n')
        # print(recurrent_states[0].shape)
        # print(recurrent_states[1].shape)
        # print(features_sequence.shape)

        # if we have resets inside the loop, manually iterate to reset the hidden states
        recurrent_output = []

        for features, episode_start in zip_strict(features_sequence, episode_starts):
            # print(features.shape, episode_start.shape)
            hidden, recurrent_states = self.agent.rssm(
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * recurrent_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * recurrent_states[1],
                ),
                features.unsqueeze(dim=0),
            )
            recurrent_output += [hidden]

        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        recurrent_output = torch.flatten(torch.cat(recurrent_output).transpose(0, 1), start_dim=0, end_dim=1)
        return recurrent_output, recurrent_states


    def forward(
        self,
        obs: torch.Tensor,
        recurrent_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)
        """
        # note due to backwards compatibility the recurrent state RNNStates is a tuple of identical states, both of which serve as the hidden states for actor/ critic
        # at each update, detach and copy over the first tuple to the second
        # shape of each tensor in the tuple is (num_lstm_layers, num_envs, lstm_hidden_size) -> so squeeze dim=0 when concat

        features = self.feature_extractor(obs)
        recurrent_output, new_recurrent_states = self._process_recurrent_states(features, recurrent_states.pi, episode_starts)

        actions, dist, logprobs, quality_logits, values = self.agent.forward_step(features, recurrent_output)
        actions = actions.reshape((-1, *self.action_space.shape))

        new_recurrent_states = RNNStates(new_recurrent_states, (new_recurrent_states[0].detach(), new_recurrent_states[1].detach()))

        return actions, values, logprobs, new_recurrent_states

    
    def predict_values(
        self,
        terminal_obs: torch.Tensor,
        terminal_recurrent_state: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor: #TODO
        """
        Predict values for the terminal observation and recurrent state.
        """
        features = self.feature_extractor(terminal_obs)
        terminal_recurrent_state = self._process_recurrent_states(features, terminal_recurrent_state, episode_starts)

        h, c = terminal_recurrent_state
        values = self.agent.get_values(features, h)
        return values

    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        recurrent_states: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Evaluate actions according to the current policy
        """
        features = self.feature_extractor(obs)
        # print(features.shape)
        recurrent_output, new_recurrent_states = self._process_recurrent_states(features, recurrent_states.pi, episode_starts)
        # print(recurrent_output.shape, new_recurrent_states[0].shape)

        _, dist, logprobs, _, values = self.agent.forward_step(features, recurrent_output)
        return values, logprobs, dist.entropy()


    def _predict(
        self,
        obs: torch.Tensor,
        recurrent_state: torch.Tensor,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor]:
        """
        Get the action from the current policy for a given observation
        """
        features = self.feature_extractor(obs)
        recurrent_state = self._process_recurrent_states(features, recurrent_state, episode_starts)

        actions, new_recurrent_state = self.agent.predict_action(features, recurrent_state.pi)
        new_recurrent_state = RNNStates(new_recurrent_state, (new_recurrent_state[0].detach(), new_recurrent_state[1].detach()))
        return actions, new_recurrent_state



class CustomRecurrentPPO(RecurrentPPO):
    """
    recurrent PPO from sb3 with some tweaks
    """
    def __init__(self, agent_class=None, *args, **kwargs):
        policy_kwargs = kwargs.get('policy_kwargs', {})
        
        agent_kwargs = kwargs.pop('agent_kwargs', {})
        
        if agent_class is not None:
            policy_kwargs['agent_class'] = agent_class
            
        for key in ['input_channels', 'action_dim', 'num_input_voltages']:
            if key in policy_kwargs:
                if 'agent_kwargs' not in policy_kwargs:
                    policy_kwargs['agent_kwargs'] = {}
                policy_kwargs['agent_kwargs'][key] = policy_kwargs.pop(key)
        
        if agent_kwargs:
            if 'agent_kwargs' not in policy_kwargs:
                policy_kwargs['agent_kwargs'] = {}
            policy_kwargs['agent_kwargs'].update(agent_kwargs)
            
        kwargs['policy_kwargs'] = policy_kwargs
        
        if agent_class is not None:
            self.policy_class = partial(CustomAgentPolicy, agent_class=agent_class, agent_kwargs=agent_kwargs)
        super().__init__(*args, **kwargs)


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            device=self.device,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        single_recurrent_shape = (self.policy.agent.rssm.num_layers, self.n_envs, self.policy.agent.rssm.hidden_dim)
        recurrent_buffer_shape = (self.n_steps,) + single_recurrent_shape

        self.rollout_buffer = RecurrentDictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            recurrent_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # self._last_lstm_states = torch.zeros(single_recurrent_shape, device=self.device) # single recurrent state for both actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_recurrent_shape, device=self.device),
                th.zeros(single_recurrent_shape, device=self.device),
            ),
            (
                th.zeros(single_recurrent_shape, device=self.device),
                th.zeros(single_recurrent_shape, device=self.device),
            ),
        )

        # initialise schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
