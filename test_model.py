import argparse
import numpy as np
import pathlib

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from roomba_env import RoombaEnv
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser(description="Test the Lunar Lander model.")
    parser.add_argument('--filename', type=str, help="Saved model", required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env = RoombaEnv(render_mode="rgb_array")
    n_actions = env.action_space.n
    # TODO: Make a factory depending on model
    model = PPO.load(args.filename)
    env._max_episode_steps = 1000
    avg_reward=0

    # test_model(model, env)
    n_steps=10
    directory= pathlib.Path("test_results/")
    directory.mkdir(exist_ok=True)
    for i_episode in range(n_steps):
        video_output_file = directory / f"test_{i_episode}.mp4"
        video_recorder = VideoRecorder(env, enabled=True, path=str(video_output_file))
        observation = env.reset()
        total_reward = 0
        done = False
        t=0
        while not done:
            env.render()
            # TODO : replace with debug() print statements
            #print("State ", observation)
            video_recorder.capture_frame()
            chosen_action, _ = model.predict(np.array([observation]))
            action = chosen_action[0]
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
                break
        avg_reward += total_reward
        video_recorder.close()
    print(f"Average reward: {avg_reward/n_steps}")
    env.close()
