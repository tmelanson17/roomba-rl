from stable_baselines3 import PPO, DQN

def create_model(model_type, env, **args):
    if model_type == "PPO":
        return PPO(policy='MlpPolicy', env=env, verbose=True, **args)
    elif model_type == "DQN":
        return DQN(policy='MlpPolicy', env=env, verbose=True, **args)

def load_model(model_type, model_filename, env=None):
    if model_type == "PPO":
        return PPO.load(model_filename, env)
    elif model_type == "DQN":
        return DQN.load(model_filename, env)
    else:
        raise ValueError("model type unknown")
