import argparse
import numpy as np
import pathlib

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from model_utils import load_model
from roomba_env import RoombaEnvAToB
from create_roomba_env import add_roomba_args, RoombaEnvFactory
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
      VecEnv,
      VecVideoRecorder,
 )

def parse_args():
    parser = argparse.ArgumentParser(description="Test the Lunar Lander model.")
    parser.add_argument('--filename', type=str, help="Saved model", required=True)
    parser.add_argument('--model', type=str, help="model type", default='ppo')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--policy', type=str, default="CnnPolicy")

    add_roomba_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    factory = RoombaEnvFactory(args)
    vec_env = make_vec_env(factory.create_roomba_env_func(), n_envs=1)
    directory= pathlib.Path("test_results/")
    directory.mkdir(exist_ok=True)
    n_actions = vec_env.action_space.n
    # TODO: Make a factory depending on model
    model = load_model(args.model.upper(), args.filename,env=vec_env)
    avg_reward=0

    # test_model(model, env)
    for i_episode in range(args.runs):
        video_output_file = directory / f"test_{i_episode}.mp4"
        env = VecVideoRecorder(vec_env, str(video_output_file), lambda x: x==0)
        observation = env.reset()
        total_reward = 0
        done = False
        t=0
        while not done:
            env.render()
            # TODO : replace with debug() print statements
            #print("State ", observation)
            lstm_states = None
            # video_recorder.capture_frame()
            # chosen_action, _ = model.predict(np.array([observation]))
            # action = chosen_action[0]
            action, lstm_states = model.predict(
                    observation,
                    state=lstm_states,
                    episode_start=done,
                    deterministic=True,
            )
            # if np.random.rand() < 0.1:
            #     action = np.random.randint(n_actions)
            # else:
            #     action = np.argmax(q_values)
            #print("Action; ", action)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Reward=", total_reward)
                env.reset()
                break
        env.close()
        avg_reward += total_reward
        # video_recorder.close()
    print(f"Average reward: {avg_reward/args.runs}")
    # env.close()
