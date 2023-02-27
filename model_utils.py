from stable_baselines3 import PPO

def create_model(model_type, env, *args):
    if model_type == "PPO":
        return PPO(policy='MlpPolicy', env=env, verbose=True, *args)

def load_model(model_type, model_filename):
    if model_type == "PPO":
        return PPO.load(model_filename)
