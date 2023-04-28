from stable_baselines3 import PPO, DQN
from imitation.algorithms import bc


def create_model(model_type, env, policy, **kwargs):
    if model_type == "PPO":
        return PPO(policy=policy, env=env, verbose=True, **kwargs)
    elif model_type == "DQN":
        return DQN(policy=policy, env=env, verbose=True, **kwargs)

def load_model(model_type, model_filename, env=None, policy="MlpPolicy", **kwargs):
    if model_type == "PPO":
        return PPO.load(model_filename, env)
    elif model_type == "DQN":
        return DQN.load(model_filename, env)
    elif model_type == "BC":
        ppo = PPO(policy=policy, env=env, verbose=True, **kwargs)
        ppo.policy = bc.reconstruct_policy(model_filename).to(ppo.device)
        return ppo
    else:
        raise ValueError("model type unknown")
