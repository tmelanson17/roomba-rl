import gym
from gym.error import DependencyNotInstalled
from gym import spaces
import numpy as np
from roomba import *
from particle import Pose, ParticleMap
from sensor import Sensor
import math
from angle_math import compute_angle_diff
import random
from dataclasses import dataclass



try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )
FPS=10
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
P_SCALE = 10.0

VIEWPORT_W = 600
VIEWPORT_H = 400

LINEAR_SPEED=15
ROTATIONAL_SPEED=1.0

N_PARTICLES=100
PARTICLE_SPEED=1

SENSOR_DETECTION_THRESHOLD=100

COLLISION_DIST=15

@dataclass
class RoombaEnvConfig():
    n_particles: int = N_PARTICLES
    particle_speed: int = PARTICLE_SPEED
    viewport_width: int = VIEWPORT_W
    viewport_height: int = VIEWPORT_H
    sensor_detection_threshold: int = SENSOR_DETECTION_THRESHOLD
    collision_dist: int = COLLISION_DIST
    fuel_cost: float = 0.01

def rotate(dp,theta):
    dx, dy = dp
    return (
        dx*math.cos(theta) - dy*math.sin(theta),
        dx*math.sin(theta) + dy*math.cos(theta)
    )



class RoombaEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array", "human"],
        "render_fps": FPS,
    }

    def _init_states(self, config: RoombaEnvConfig) -> None:
        roomba_start_x = config.viewport_width // 2
        roomba_start_y = config.viewport_height // 2
        self._roomba = Roomba(
            pos=Pose(
                x=roomba_start_x,
                y=roomba_start_y,
                theta=0.
            ), 
            dx=LINEAR_SPEED/FPS,
            dtheta=ROTATIONAL_SPEED/FPS
        )
        roomba_buffer=5
        free_space = (
            (roomba_start_x - roomba_buffer, roomba_start_y - roomba_buffer),
            (roomba_start_x + roomba_buffer, roomba_start_y + roomba_buffer),
        )
        self._particles = ParticleMap(
            config.n_particles,
            config.viewport_width,
            config.viewport_height, 
            free_space=free_space,
            max_dist=config.particle_speed,
            collision_dist=config.collision_dist
        )
        self._sensor = Sensor(SENSOR_DETECTION_THRESHOLD)
        self._i = 0
        self._bounds = (VIEWPORT_W, VIEWPORT_H)


    def __init__(self, roomba_env_config=None, render_mode="rgb_array", max_episode_steps=1000) -> None:
        super().__init__()
        if roomba_env_config is None:
            roomba_env_config = RoombaEnvConfig()
        self.action_space = spaces.Discrete(4)
        low = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        high = np.array([1.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self._init_states(roomba_env_config)
        self._max_episode_steps = 200
        self.render_mode = render_mode
        self.screen: pygame.Surface = None
        self.clock = None
        self.terminated = False
        
        
    def step(self, action):
        # TODO: clean this up
        if self._i >= self._max_episode_steps:
            self.terminated = True
        if self.terminated:
            sensor_output = self._sensor.sense(self._roomba, self._particles)
            return np.array(sensor_output, dtype=np.float32), 0, self.terminated, {}
        self._roomba.move(action, self._bounds)
        self._particles.move()
        reward = 0
        if self._particles.detect_collision(self._roomba.pose):
            reward = -100
            self.terminated = True
        # If moving forward, gain point
        elif action == 0:
            reward = 1
        # If moving backwards, then penalize
        elif action == 2:
            reward = -2
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        self._i += 1
        return np.array(sensor_output, dtype=np.float32), reward, self.terminated, {}

    # Need for the initial state
    # TODO : implement seed
    def reset(self, seed=0):
        self._init_states()
        self.terminated = False
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        return np.array(sensor_output, dtype=np.float32)


    def render(self, mode=None):
        # render_mode = mode if mode else self.render_mode
        render_mode = self.render_mode
        if self.screen is None and render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (int(SCALE), int(SCALE)))


        # Draw roomba
        roomba_pose = self._roomba.pose
        x = int(roomba_pose.x) 
        y = int(roomba_pose.y) 
        dps = [
            (2, 0),
            (6, 6),
            (0, 8),
            (-6, 6),
            (-8, 0),
            (-6, -6),
            (0, -8),
            (6, -6)
        ]
        rotated_dp = [rotate(p, roomba_pose.theta) for p in dps]
        points = [
            (int(x + p[0]), int(y + p[1])) for p in rotated_dp
        ]
        pygame.draw.polygon(
            self.surf,
            (255, 255, 255), # color
            points # points
        )

        # Draw sensor output
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        for sensor_level, angle_sensor in zip(sensor_output, self._sensor.angles):
            angle_global = angle_sensor + roomba_pose.theta
            sensor_raw_dist = sensor_level * self._sensor.detection_threshold
            pygame.draw.line(
                    self.surf,
                    (0, 255, 255), # color
                    (x, y), # start_pos
                    (
                        int(x + sensor_raw_dist*math.cos(angle_global)), 
                        int(y + sensor_raw_dist*math.sin(angle_global)),
                    ), # end_pos
            )

        # Draw objects
        for pos in self._particles.particles:
            if pos.x < 0 or pos.y < 0:
                continue
            if pos.x > VIEWPORT_W or pos.y > VIEWPORT_H:
                continue
            pygame.draw.circle(
                self.surf,
                (255,0,0), # color
                (int(pos.x), int(pos.y)), # center
                2, # radius
            )
        
        if render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


class RoombaEnvAToB(gym.Env):
    metadata = {
        "render.modes": ["rgb_array", "human"],
        "render_fps": FPS,
    }

    def _init_states(self, seed=0):
        self._rnd = random.Random()
        # Hardcode the goal
        self.goal = (
                3*self.config.viewport_width/4,
                3*self.config.viewport_height/4,
        )
        # self.goal = (
        #         self._rnd.random()*self.config.viewport_width, 
        #         self._rnd.random()*self.config.viewport_height, 
        # )
        roomba_start_x = self.config.viewport_width // 2
        roomba_start_y = self.config.viewport_height // 2
        self._roomba = Roomba(
            pos=Pose(
                x=roomba_start_x,
                y=roomba_start_y,
                theta=0.
            ), 
            dx=LINEAR_SPEED/FPS,
            dtheta=ROTATIONAL_SPEED/FPS
        )
        self.terminated = False
        # TODO: Make this a config param
        roomba_buffer=50
        free_space = (
            (roomba_start_x - roomba_buffer, roomba_start_y - roomba_buffer),
            (roomba_start_x + roomba_buffer, roomba_start_y + roomba_buffer),
        )
        self._particles = ParticleMap(
            self.config.n_particles,
            self.config.viewport_width,
            self.config.viewport_height, 
            free_space=free_space,
            max_dist=self.config.particle_speed,
            collision_dist=self.config.collision_dist
        )
        self._sensor = Sensor(SENSOR_DETECTION_THRESHOLD)
        self._i = 0
        self._bounds = (VIEWPORT_W, VIEWPORT_H)
        self._last_obs = self.measure()
        return self._last_obs
        
    def __init__(self, roomba_env_config=None, render_mode="rgb_array", max_episode_steps=1000) -> None:
        super().__init__()
        if roomba_env_config is None:
            roomba_env_config = RoombaEnvConfig()
        self.config = roomba_env_config
        self.action_space = spaces.Discrete(4)
        # Observation space: (x, y, theta,) of goal
        low = [0, 0]
        high = [self.config.viewport_width**2+self.config.viewport_height**2, math.pi]
        # Observation space: (s1, s2, s3) of sensor output
        low += [0.0, 0.0, 0.0]
        high += [100.0, 100.0, 100.0]
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        self.observation_space = spaces.Box(low, high)
        self._init_states()
        self._max_episode_steps = 200
        self.render_mode = render_mode
        self.screen: pygame.Surface = None
        self.clock = None

    # TODO:Move to sensor
    def measure(self):
        roomba_x, roomba_y, roomba_theta = self._roomba.pose
        goal_x, goal_y = self.goal
        dx = goal_x - roomba_x
        dy = goal_y - roomba_y
        # Calculate angle of goal to roomba

        goal_theta = math.atan2(dy, dx)
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        sensor_output = tuple(s*100 for s in sensor_output)
        diff = compute_angle_diff(goal_theta, roomba_theta)
        obs = (
                math.sqrt((goal_x - roomba_x)**2 + (goal_y - roomba_y)**2),
                diff, 
        ) + sensor_output
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self, obs):
        d_theta = obs[1]
        # TODO: Make this a parameter
        OPPOSITE_THRESHOLD = math.pi/2
        CLOSE_THRESHOLD = math.pi/8
        # Three-tier : -1 if opposite direction, 1 if very close
        if d_theta > OPPOSITE_THRESHOLD:
            theta_reward = -2
        elif d_theta < CLOSE_THRESHOLD:
            theta_reward = 1
        else:
            theta_reward = 0
        distance = obs[0]
        last_distance = self._last_obs[0]
        # You reached the goal!
        if distance < self.config.collision_dist:
            self.terminated = True
            return 100
        return (
                theta_reward +
                (last_distance - distance)
                - self.config.fuel_cost
        )

    def step(self, action):
        # Reward is based on distance
        if self.terminated:
            return None, 0, self.terminated, {}
        self._roomba.move(action, self._bounds)
        self._particles.move()
        if self._i >= self._max_episode_steps:
            self.terminated = True
        obs = self.measure()
        reward = self.calculate_reward(obs)
        # print(f"Reward: {reward}")
        # You hit the particle :(
        self._i += 1
        self._last_obs = obs
        if self._particles.detect_collision(self._roomba.pose):
            reward = -200
            self.terminated = True
            obs = None
        return obs, reward, self.terminated, {}

    def reset(self, seed=0):
        return self._init_states(0)

    def render(self, mode=None):
        # render_mode = mode if mode else self.render_mode
        render_mode = self.render_mode
        if self.screen is None and render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (int(SCALE), int(SCALE)))

        # Draw roomba
        roomba_pose = self._roomba.pose
        x = int(roomba_pose.x) 
        y = int(roomba_pose.y) 
        dps = [
            (2, 0),
            (6, 6),
            (0, 8),
            (-6, 6),
            (-8, 0),
            (-6, -6),
            (0, -8),
            (6, -6)
        ]
        rotated_dp = [rotate(p, roomba_pose.theta) for p in dps]
        points = [
            (int(x + p[0]), int(y + p[1])) for p in rotated_dp
        ]
        pygame.draw.polygon(
            self.surf,
            (255, 255, 255), # color
            points # points
        )

        # Draw goal as crosshair
        CROSSHAIR_LENGTH = 3
        pygame.draw.line(
            self.surf,
            (255, 255, 0), # color
            (int(self.goal[0] - CROSSHAIR_LENGTH), int(self.goal[1])), # start_pos
            (int(self.goal[0] + CROSSHAIR_LENGTH), int(self.goal[1])), # end_pos
            width=10,
        )
        pygame.draw.line(
            self.surf,
            (255, 255, 0), # color
            (int(self.goal[0]), int(self.goal[1] - CROSSHAIR_LENGTH)), # start_pos
            (int(self.goal[0]), int(self.goal[1] + CROSSHAIR_LENGTH)), # end_pos
            width=10,
        )

        # Draw sensor output
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        for sensor_level, angle_sensor in zip(sensor_output, self._sensor.angles):
            angle_global = angle_sensor + roomba_pose.theta
            sensor_raw_dist = sensor_level * self._sensor.detection_threshold
            pygame.draw.line(
                    self.surf,
                    (0, 255, 255), # color
                    (x, y), # start_pos
                    (
                        int(x + sensor_raw_dist*math.cos(angle_global)), 
                        int(y + sensor_raw_dist*math.sin(angle_global)),
                    ), # end_pos
            )

        # Draw objects
        for pos in self._particles.particles:
            if pos.x < 0 or pos.y < 0:
                continue
            if pos.x > VIEWPORT_W or pos.y > VIEWPORT_H:
                continue
            pygame.draw.circle(
                self.surf,
                (255,0,0), # color
                (int(pos.x), int(pos.y)), # center
                2, # radius
            )
        
        if render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


if __name__ == '__main__':
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    # Stress test
    N_PARTICLES=1000
    SPEED=5
    env = RoombaEnv(render_mode="rgb_array")
    video_recorder = VideoRecorder(env, enabled=True, path='random_actions.mp4')
    for i in range(100):
        video_recorder.capture_frame()
        state, reward, terminated, _ = env.step(env.action_space.sample())
        print(state)
        print(reward)
        print("===========")
        if terminated:
            break
    print(video_recorder.path)
    video_recorder.close()

    # Test the new env
    env_a_to_b = RoombaEnvAToB(render_mode="rgb_array")
    video_recorder_a_to_b = VideoRecorder(env_a_to_b, enabled=True, path='a_to_b.mp4')
    for i in range(4):
        for j in range(25):
            video_recorder_a_to_b.capture_frame()
            state, reward, terminated, _ = env_a_to_b.step(i)
            print(state)
            print(reward)
            print("===========")
        if terminated:
            break
    print(video_recorder_a_to_b.path)
    video_recorder_a_to_b.close()

