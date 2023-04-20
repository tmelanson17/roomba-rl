from functools import partial
from roomba_env import RoombaEnvAToB, RoombaEnvConfig

def add_roomba_args(parser):
    # Roomba env arguments
    parser.add_argument('--n-particles', type=int, default=100, required=False)
    parser.add_argument('--hardcode-map', action="store_true", required=False)
    parser.add_argument('--goal', type=int, nargs="+", required=False, default=None)
    parser.add_argument('--roomba-speed', type=int, default=None)
    parser.add_argument('--human', action="store_true", required=False)
    parser.add_argument('--disable-visible-particles', action="store_true", required=False)
    parser.add_argument('--max-episode-steps', type=int, default=200, required=False)
    parser.add_argument('--fuel-cost', type=float, default=0.01, required=False)

class RoombaEnvFactory:
    def __init__(self, args): 
        self.roomba_env_config = RoombaEnvConfig()
        self.roomba_env_config.n_particles = args.n_particles
        self.roomba_env_config.hardcode_particle_map = args.hardcode_map
        if args.roomba_speed is not None:
            self.roomba_env_config.linear_speed *= args.roomba_speed
            self.roomba_env_config.rotational_speed *= args.roomba_speed
        if args.disable_visible_particles:
            self.roomba_env_config.visible_particles = False
        self.roomba_env_config.goal = None if args.goal is None else tuple(args.goal)
        self.roomba_env_config.fuel_cost = args.fuel_cost
        self.render_mode = "human" if args.human else "rgb_array"
        self.max_episode_steps = args.max_episode_steps

    def create_roomba_env(self):
        return RoombaEnvAToB(roomba_env_config=self.roomba_env_config, render_mode=self.render_mode, max_episode_steps=self.max_episode_steps)

    def create_roomba_env_func(self):
        return self.create_roomba_env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Roomba model.")
    args = parser.parse_args()
    factory = RoombaEnvFactory(args)

    roomba_env_func = factory.create_roomba_env_func()

    roomba_env_func()

