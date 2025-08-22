from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union, List
from dataclasses import dataclass
from tqdm import tqdm
import wandb
import numpy as np
import torch as th
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")

try:
    from custom_policy import RecurrentActorCriticPolicy
    from custom_feature_extractor import CustomFeatureExtractor
except ModuleNotFoundError:
    from policy.custom_policy import RecurrentActorCriticPolicy
    from policy.custom_feature_extractor import CustomFeatureExtractor


@dataclass
class MultiAgentSetup:
    num_dots: int
    gate_voltage_range: List[float]
    barrier_voltage_range: List[float]
    obs_image_size: int = 128
    gate_obs_image_channels: int = 2
    barrier_obs_image_channels: int = 1
    num_agent_voltages: int = 1 # both obs and action

    

class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_recurrent_policies`
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "CustomRecurrentPolicy": RecurrentActorCriticPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        use_wandb: bool = False,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Dict,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.optimizer_kwargs = optimizer_kwargs
        self._last_lstm_states = None

        self.use_wandb = use_wandb

        self.multi_agent_setup = MultiAgentSetup(
            num_dots=4,
            gate_voltage_range=[-10.0, 2.0],
            barrier_voltage_range=[-2.0, 2.0] # TODO placeholder for now
        )
        self.num_plungers = self.multi_agent_setup.num_dots
        self.num_barriers = self.num_plungers - 1

        # for mean reward logging
        self.total_reward = 0.0
        self.reward_count = 0

        if use_wandb:
            wandb.init(
                project="recurrent-ppo",
                config={
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_range": clip_range,
                    "ent_coef": ent_coef,
                    "vf_coef": vf_coef,
                    "policy_kwargs": policy_kwargs,
                }
            )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentRolloutBuffer

        features_extractor_class = CustomFeatureExtractor
        features_extractor_kwargs = {
            "features_dim": 256,
            "voltage_dim": 1,
        }

        image_size = self.multi_agent_setup.obs_image_size
        gate_voltage_range = self.multi_agent_setup.gate_voltage_range
        barrier_voltage_range = self.multi_agent_setup.barrier_voltage_range

        gate_observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(2, image_size, image_size), dtype=np.uint8
            ),
            "obs_voltages": spaces.Box(
                low=gate_voltage_range[0], high=gate_voltage_range[1], shape=(1,), dtype=np.float32
            )
        })

        barrier_observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, shape=(1, image_size, image_size), dtype=np.uint8
            ),
            "obs_voltages": spaces.Box(
                low=barrier_voltage_range[0], high=barrier_voltage_range[1], shape=(1,), dtype=np.float32
            )
        })

        self.gate_action_space = spaces.Box(
            low=gate_voltage_range[0],
            high=gate_voltage_range[1],
            shape=(self.num_plungers,),
            dtype=np.float32
        )

        self.barrier_action_space = spaces.Box(
            low=barrier_voltage_range[0],
            high=barrier_voltage_range[1],
            shape=(self.num_barriers,),
            dtype=np.float32
        )

        policy_init_kwargs = {
            "lr_schedule": self.lr_schedule,
            "use_sde": self.use_sde,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
            #"shared_lstm": True,
            **self.policy_kwargs,
        }

        self.gate_policy = self.policy_class(observation_space=gate_observation_space, action_space=self.gate_action_space, **policy_init_kwargs)
        self.gate_policy = self.gate_policy.to(self.device)

        self.barrier_policy = self.policy_class(observation_space=barrier_observation_space, action_space=self.barrier_action_space, **policy_init_kwargs)
        self.barrier_policy = self.barrier_policy.to(self.device)

        self.policy = None # ensure we are not using the default policy


        # set up one central optimizer for all the policies
        optimizer_kwargs = {} if self.optimizer_kwargs is None else self.optimizer_kwargs
        self.gate_optimizer = th.optim.Adam(self.gate_policy.parameters(), lr=self.lr_schedule(1), **optimizer_kwargs)
        self.barrier_optimizer = th.optim.Adam(self.barrier_policy.parameters(), lr=self.lr_schedule(1), **optimizer_kwargs)

        # We assume that LSTM for the actor and the critic have the same architecture
        lstm = self.gate_policy.lstm_actor

        if not isinstance(self.gate_policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)

        # hidden and cell states for actor and critic for each agent
        last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        self._last_lstm_states = {
            "gates": [last_lstm_states] * self.num_plungers,
            "barriers": [last_lstm_states] * self.num_barriers,
        }

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)


        # TODO may need to assign a hidden state per agent to the buffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.gate_policy.set_training_mode(False)
        self.barrier_policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.gate_policy.reset_noise(env.num_envs)
            self.barrier_policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.gate_policy.reset_noise(env.num_envs)
                self.barrier_policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)

                # first split the images and voltages and pass each through each agent's feature extractor

                image_obs = obs_tensor["image"]
                gate_voltage_obs = obs_tensor["obs_voltages"].flatten()

                gate_observations = []
                barrier_observations = []

                first_img = image_obs[:,0,:,:]
                gate_observations.append({
                    "image": th.stack([first_img, first_img], dim=1),
                    "obs_voltages": gate_voltage_obs[0],
                })

                for i in range(self.num_plungers-2):
                    v = gate_voltage_obs[i+1]
                    img = image_obs[:,i:i+2,:,:]
                    gate_observations.append({
                        "image": img,
                        "obs_voltages": v
                    })

                last_img = image_obs[:,-1,:,:]
                gate_observations.append({
                    "image": th.stack([last_img, last_img], dim=1),
                    "obs_voltages": gate_voltage_obs[-1],
                })


                for i in range(self.num_barriers):
                    barrier_observations.append({
                        "image": image_obs[:,i,:,:],
                        "obs_voltages": torch.tensor([0], dtype=torch.float32), # TODO we will feed in the barrier voltages to the barrier agents, once qarray has this functionality
                    })


                gate_actions = [] # list of voltages of size self.num_plungers
                gate_values = []
                gate_lstm_states = []
                gate_log_probs = []
                for obs, gate_state in zip(gate_observations, lstm_states["gates"]):
                    actions, values, log_probs, gate_state = self.gate_policy.forward(obs, gate_state, episode_starts)
                    gate_actions.append(actions)
                    gate_values.append(values)
                    gate_lstm_states.append(gate_state)
                    gate_log_probs.append(log_probs)

                barrier_actions = [] # list of voltages of size self.num_barriers
                barrier_values = []
                barrier_lstm_states = []
                barrier_log_probs = []
                for obs, barrier_state in zip(barrier_observations, lstm_states["barriers"]):
                    actions, values, log_probs, barrier_state = self.barrier_policy.forward(obs, barrier_state, episode_starts)
                    barrier_actions.append(actions)
                    barrier_values.append(values)
                    barrier_lstm_states.append(barrier_state)
                    barrier_log_probs.append(log_probs)

                # actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

            # TODO check handling of values in training since original code has tensor of shape (num_envs, 1) -> is this the most efficient way to do this
            # note that first dim is n envs
            gate_values = torch.stack(gate_values).swapaxes(0, 1) # (num_envs, num_gates, 1)
            barrier_values = torch.stack(barrier_values).swapaxes(0, 1) # (num_envs, num_barriers, 1)

            all_values = []
            for i in range(gate_values.shape(0)):
                all_values.append({
                    "gates": gate_values[i],
                    "barriers": barrier_values[i]
                })

            gate_log_probs = torch.stack(gate_log_probs).swapaxes(0, 1) # (num_envs, num_gates, 1)
            barrier_log_probs = torch.stack(barrier_log_probs).swapaxes(0, 1) # (num_envs, num_barriers, 1)

            all_log_probs = []
            for i in range(gate_log_probs.shape(0)):
                all_log_probs.append({
                    "gates": gate_log_probs[i],
                    "barriers": barrier_log_probs[i]
                })

            lstm_states = {
                "gates": gate_lstm_states,
                "barriers": barrier_lstm_states
            }

            # actions = actions.cpu().numpy()
            gate_actions = th.stack(gate_actions).cpu().numpy()
            barrier_actions = th.stack(barrier_actions).cpu().numpy()

            # Rescale and perform action
            clipped_gate_actions = gate_actions
            clipped_barrier_actions = barrier_actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.gate_action_space, spaces.Box):
                clipped_gate_actions = np.clip(clipped_gate_actions, self.gate_action_space.low, self.gate_action_space.high)
            if isinstance(self.barrier_action_space, spaces.Box):
                clipped_barrier_actions = np.clip(clipped_barrier_actions, self.barrier_action_space.low, self.barrier_action_space.high)

            action_dict = {
                "gates": clipped_gate_actions,
                "barriers": clipped_barrier_actions,
            }

            # note all of these are arrays with the first dim = num of envs
            new_obs, reward_dict, dones, infos = env.step(action_dict) # TODO let env accept an action dict and return a reward dict for the barriers and gates


            # add the raw observation and reward dictionaries to the buffer, only separate during training

            if isinstance(reward_dict["gates"], np.array):
                rewards = reward_dict["gates"].flatten().tolist() + reward_dict["barriers"].flatten().tolist()
            else:
                rewards = reward_dict["gates"] + reward_dict["barriers"]
                
            if self.use_wandb:
                wandb.log({
                    "reward": np.mean(rewards),
                })

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    
                    # Original code:
                    # terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    # with th.no_grad():
                    #     terminal_lstm_state = (
                    #         lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                    #         lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                    #     )
                    #     # terminal_lstm_state = None
                    #     episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                    #     terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    # rewards[idx] += self.gamma * terminal_value

                    # use hidden value function states to bootstrap the value function

                    # TODO check this loop

                    with torch.no_grad():
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)

                        full_image_obs = infos[idx]["terminal_observation"]["image"]
                        full_gate_voltage_obs = infos[idx]["terminal_observation"]["obs_voltages"].flatten()

                        for g in range(self.num_plungers):
                            terminal_lstm_state = (
                                lstm_states["gates"][g].vf[0][:, idx : idx + 1, :].contiguous(),
                                lstm_states["gates"][g].vf[1][:, idx : idx + 1, :].contiguous(),
                            )

                            if g == 0:
                                image_obs = torch.tensor(full_image_obs[:,0,:,:])
                                image_obs = torch.cat([image_obs]*2, dim=1)
                            elif g == self.num_plungers - 1:
                                image_obs = torch.tensor(full_image_obs[:,-1,:,:])
                                image_obs = torch.cat([image_obs]*2, dim=1)
                            else:
                                image_obs = torch.tensor(full_image_obs[:,g-1:g+1,:,:])
                            
                            terminal_gate_obs = {
                                "image": image_obs,
                                "obs_voltages": full_gate_voltage_obs[g],

                            }
                            terminal_value = self.gate_policy.predict_values(terminal_gate_obs, terminal_lstm_state, episode_starts)[0]
                            reward_dict[idx]["gates"][g] += self.gamma * terminal_value

                        for b in range(self.num_barriers):
                            terminal_lstm_state = (
                                lstm_states["barriers"][b].vf[0][:, idx : idx + 1, :].contiguous(),
                                lstm_states["barriers"][b].vf[1][:, idx : idx + 1, :].contiguous(),
                            )
                            
                            terminal_barrier_obs = {
                                "image": torch.tensor(full_image_obs[:,b,:,:]),
                                "obs_voltages": torch.zeros(1, dtype=torch.float32) # TODO add barrier voltages
                            }
                            terminal_value = self.barrier_policy.predict_values(terminal_barrier_obs, terminal_lstm_state, episode_starts)[0]
                            reward_dict[idx]["barriers"][b] += self.gamma * terminal_value


            rollout_buffer.add(
                self._last_obs,
                action_dict,
                reward_dict,
                self._last_episode_starts,
                all_values,
                all_log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        #######
        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            for g in range(self.num_plungers):
                pass
                
            for b in range(self.num_barriers):
                pass
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/scaled_entropy_loss": entropy_loss.item() * self.ent_coef,
                        "train/policy_loss": policy_loss.item(),
                        "train/scaled_value_loss": value_loss.item() * self.vf_coef,
                    })

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"]
