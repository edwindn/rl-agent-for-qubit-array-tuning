import torch

from envs.qarray_ndot_env import QuantumDeviceEnv
from model.voltage_agent import Agent
from policy.ppo_recurrent import RecurrentPPO
from policy.custom_policy import RecurrentActorCriticPolicy

from stable_baselines3.common.env_util import make_vec_env


# def main0):
#     env = QuantumDeviceEnv()
#
#     model = CustomRecurrentPPO(
#         agent_class=Agent,\
#         agent_kwargs={
#             'input_channels': 1,
#             'action_dim': 2,
#             'num_input_voltages': 2,
#         },
#         policy=CustomAgentPolicy,
#         env=env,
#         use_wandb=True,
#         learning_rate=1e-4,
#         gamma=0.99,
#         ent_coef=1e-3,
#         vf_coef=5e-4,
#         gae_lambda=0.95,
#     )
    

def main():
    env = make_vec_env(QuantumDeviceEnv, n_envs=1)
    model = RecurrentPPO(
        "CustomRecurrentPolicy",
        env,
        verbose=1,
        use_wandb=False,
        vf_coef=1e-5,
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("recurrent_ppo_v0")


if __name__ == '__main__':
    main()