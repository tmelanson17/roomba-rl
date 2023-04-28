
from time import time
import numpy as np
import copy
import argparse
import sys
from roomba_env import RoombaEnvAToB
from model_utils import create_model, load_model
from create_roomba_env import add_roomba_args, RoombaEnvFactory

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from huggingface_sb3 import package_to_hub


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Roomba model.")
    parser.add_argument('--episodes', type=int, default=300000, required=False)
    parser.add_argument('--model', type=str, default="ppo")

    # PPO args
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--epsilon', type=float, default=0.2)


    parser.add_argument('--eval-only', action="store_true", default=False)
    parser.add_argument('--train-only', action="store_true", default=False)
    parser.add_argument('--preloaded-model', type=str, default=None, required=False)
    parser.add_argument('--output-tag', type=str, required=False, default="default")
    parser.add_argument('--policy', type=str, required=False, default="MlpPolicy")

    add_roomba_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    factory = RoombaEnvFactory(args)
    if args.policy == "MlpPolicy":
        env = make_vec_env(factory.create_roomba_env_func(), n_envs=4)
    else:
        env = factory.create_roomba_env()
    # env = factory.create_roomba_env_func()
    n_actions = env.action_space.n
    model_architecture = args.model.upper()
    output_file = '{}-{}'.format(model_architecture, args.output_tag) #.format(args.gamma, args.episodes, args.C, args.replay_memory_size)
    env_id = "RoombaAToB-{}".format(args.output_tag)
    if not args.eval_only:
        if args.preloaded_model is not None:
            model = load_model(model_architecture, args.preloaded_model, env, policy=args.policy, gamma=args.gamma, ent_coef=args.beta, clip_range=args.epsilon)
        else:
            model = create_model(model_architecture, env, policy=args.policy, gamma=args.gamma, gae_lambda=.5)
            #model = create_model(model_architecture, env, exploration_final_eps=0.1, exploration_fraction=0.7, buffer_size=100000)
        model.learn(args.episodes)
        model.save(output_file)
    if not args.train_only:
        if args.eval_only:
            model = load_model(model_architecture, output_file)
        model_name = output_file
        # TODO: replace culteejen with username
        repo_id = f"culteejen/{model_name}-{env_id}"
        eval_env = SubprocVecEnv([factory.create_roomba_env_func()])
        package_to_hub(
               model=model, # Our trained model
               model_name=model_name, # The name of our trained model
               model_architecture=model_architecture, # The model architecture we used: in our case PPO
               env_id=env_id, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
               commit_message="Upload model to Hugging Face"
        )
