from roomba_env import RoombaEnvAToB

from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.evaluation import evaluate_policy
from create_roomba_env import add_roomba_args, create_roomba_env

import numpy as np
import pygame

from imitation.data.types import Trajectory


def get_human_feedback():
    keypress=-1
    while keypress < 0:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    print('LEFT')
                    keypress= 3
                if event.key == pygame.K_RIGHT:
                    print('RIGHT')
                    keypress= 1
                if event.key == pygame.K_UP:
                    print('FWD')
                    keypress= 0
                if event.key == pygame.K_DOWN:
                    print('REV')
                    keypress= 2
    return keypress


def human_control_loop(env):
    observations = []
    actions = []
    rewards = []
    obs = env.reset()
    observations.append(obs)
    while not env.terminated:
        env.render()
        # Render environment
        if not env.terminated:
            # Get human feedback
            action = get_human_feedback()
            actions.append(action)
            # Use action to step
            obs, reward, terminated, _ = env.step(action)
            observations.append(obs)
            rewards.append(reward)
    return Trajectory(obs=observations, acts=actions, terminal=True, infos=None)


record_data=True
if __name__ == "__main__":
    env = create_roomba_env(args)
    import pickle

    # Writing
    if record_data: 
        trajectories = []
        for i in range(10):
            trajectories.append(human_control_loop(env))
        fileObj = open('trajectories.obj', 'wb')
        pickle.dump(trajectories,fileObj)
        fileObj.close()
    else:
        fileObj = open('trajectories.obj', 'rb')
        trajectories = pickle.load(fileObj)
    for t in trajectories:
        print(len(t))
    transitions = rollout.flatten_trajectories(trajectories)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(),
    )
    bc_trainer.train(n_epochs=10)
    reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward: {reward}")
    bc_trainer.save_policy("ppo_bc_policy")
