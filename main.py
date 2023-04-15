
from time import time
import numpy as np
import copy
import argparse
import sys
from roomba_env import RoombaEnvAToB, RoombaEnvConfig
from model_utils import create_model, load_model

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from huggingface_sb3 import package_to_hub


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Roomba model.")
    parser.add_argument('--episodes', type=int, default=300000, required=False)
    parser.add_argument('--max-episode-steps', type=int, default=200, required=False)
    parser.add_argument('--model', type=str, default="ppo")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eval-only', action="store_true", default=False)
    parser.add_argument('--train-only', action="store_true", default=False)
    parser.add_argument('--preloaded-model', type=str, default=None, required=False)
    parser.add_argument('--output-tag', type=str, required=False, default="default")

    # Roomba env arguments
    parser.add_argument('--n-particles', type=int, default=100, required=False)
    parser.add_argument('--hardcode-map', action="store_true", required=False)
    parser.add_argument('--goal', type=int, nargs="+", required=False, default=None)
    parser.add_argument('--roomba-speed', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    def create_roomba_env():
        roomba_env_config = RoombaEnvConfig()
        roomba_env_config.n_particles = args.n_particles
        roomba_env_config.hardcode_particle_map = args.hardcode_map
        if args.roomba_speed is not None:
            roomba_env_config.linear_speed *= args.roomba_speed
            roomba_env_config.rotational_speed *= args.roomba_speed
        roomba_env_config.goal = None if args.goal is None else tuple(args.goal)
        return RoombaEnvAToB(roomba_env_config=roomba_env_config, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    
    env = make_vec_env(create_roomba_env, n_envs=4)
    n_actions = env.action_space.n
    model_architecture = args.model.upper()
    output_file = '{}-{}'.format(model_architecture, args.output_tag) #.format(args.gamma, args.episodes, args.C, args.replay_memory_size)
    env_id = "RoombaAToB-{}".format(args.output_tag)
    if not args.eval_only:
        if args.preloaded_model is not None:
            model = load_model(model_architecture, args.preloaded_model, env)
        else:
            model = create_model(model_architecture, env, gamma=args.gamma, gae_lambda=.5)
            #model = create_model(model_architecture, env, exploration_final_eps=0.1, exploration_fraction=0.7, buffer_size=100000)
        model.learn(args.episodes)
        model.save(output_file)
    if not args.train_only:
        if args.eval_only:
            model = load_model(model_architecture, output_file)
        model_name = output_file
        # TODO: replace culteejen with username
        repo_id = f"culteejen/{model_name}-{env_id}"
        eval_env = SubprocVecEnv([create_roomba_env])
        package_to_hub(
               model=model, # Our trained model
               model_name=model_name, # The name of our trained model
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message="Upload model to Hugging Face"
        )
