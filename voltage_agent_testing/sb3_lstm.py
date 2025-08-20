from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from envs.qarray_env import QuantumDeviceEnv

# raw sb3 with no mods
# will need to add stopping net training and custom models


def main():
    env = make_vec_env(QuantumDeviceEnv, n_envs=1)
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        use_wandb=True,
        vf_coef=1e-5,
    )

    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("recurrent_ppo_v0")


if __name__ == '__main__':
    main()