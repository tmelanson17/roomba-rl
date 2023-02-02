
from time import time
import numpy as np
import copy
import argparse
import sys
from roomba_env import RoombaEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Lunar Lander model.")
    parser.add_argument('--gamma', type=float, default=0.99, required=False, help="Gamma value")
    parser.add_argument('--episodes', type=int, default=1500, required=False)
    parser.add_argument('--C', type=int, default=50, required=False)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--initial_epsilon', type=float, default=1.0)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--replay_memory_size', type=int, default=10000)
    return parser.parse_args()

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    def create_roomba_env(max_episode_steps=1000):
        return RoombaEnv(render_mode="rgb_array", max_episode_steps=max_episode_steps)
    
    env = make_vec_env(create_roomba_env, n_envs=4)
    env_id = "Roomba"
    n_actions = env.action_space.n
    args = parse_args()
    ppo = PPO(policy='MlpPolicy', env=env, verbose=True)
    ppo.learn(100000)
    output_file = './ppo-default' #.format(args.gamma, args.episodes, args.C, args.replay_memory_size)
    ppo.save(output_file)
    ppo.load(output_file)
    model_architecture = 'PPO'
    model = ppo
    model_name = f"{model_architecture}-default"
    # TODO: replace culteejen with username
    repo_id = f"culteejen/{model_name}-{env_id}"
    eval_env = make_vec_env(create_roomba_env, n_envs=1)
    package_to_hub(
           model=model, # Our trained model
           model_name=model_name, # The name of our trained model
           model_architecture=model_architecture, # The model architecture we used: in our case PPO
           env_id=env_id, # Name of the environment
           eval_env=eval_env, # Evaluation Environment
           repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
           commit_message="Upload model to Hugging Face"
    )
