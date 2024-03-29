import gym
from gym.error import DependencyNotInstalled
from gym import spaces
import numpy as np
from roomba import *
from particle import Pose, ParticleMap, ParticleHardcodedMap
from sensor import Sensor
import math
from angle_math import compute_angle_diff
import random
from dataclasses import dataclass
from enum import Enum


try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )

FPS=30
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
P_SCALE = 10.0

VIEWPORT_W = 600
VIEWPORT_H = 400

LINEAR_SPEED=50
ROTATIONAL_SPEED=0.5

N_PARTICLES=100
PARTICLE_SPEED=1

SENSOR_DETECTION_THRESHOLD=100

COLLISION_DIST=10

N_SENSORS=8

SENSOR_ANGLES = [i*math.pi/4 for i in range(N_SENSORS)]

SensorType = Enum('Sensor', ['ULTRASONIC', 'VISUAL'])

@dataclass
class RoombaEnvConfig():
    n_particles: int = N_PARTICLES
    hardcode_particle_map: bool = False
    visible_particles: bool = True
    goal: tuple = None
    particle_speed: int = PARTICLE_SPEED
    linear_speed: int = LINEAR_SPEED
    rotational_speed: int = ROTATIONAL_SPEED
    viewport_width: int = VIEWPORT_W
    viewport_height: int = VIEWPORT_H
    sensor_detection_threshold: int = SENSOR_DETECTION_THRESHOLD
    collision_dist: int = COLLISION_DIST
    fuel_cost: float = 0.1
    wraparound: bool = False
    observation_space: SensorType = SensorType.ULTRASONIC

def rotate(dp,theta):
    dx, dy = dp
    return (
        dx*math.cos(theta) - dy*math.sin(theta),
        dx*math.sin(theta) + dy*math.cos(theta)
    )

HARDCODED_MAP = ParticleHardcodedMap(
    pygame.surfarray.array_red(
        pygame.image.load("canvas.png")
    ),
    VIEWPORT_W,
    VIEWPORT_H, 
    max_dist=0,
    collision_dist=COLLISION_DIST,
)

class RoombaEnvAToB(gym.Env):
    metadata = {
        "render.modes": ["rgb_array", "human"],
        "render_fps": FPS,
    }

    def _init_states(self, seed=0):
        self._rnd = random.Random()
        self._surface_intialized=False
        self._rendered_current=False
        if self.config.goal is None:
            self.goal = (
                    self._rnd.random()*self.config.viewport_width, 
                    self._rnd.random()*self.config.viewport_height, 
            )
        # Hardcode the goal
        else:
            self.goal = self.config.goal
        roomba_start_x = self.config.viewport_width // 2 - 20
        roomba_start_y = self.config.viewport_height // 2 - 20
        self._roomba = Roomba(
            pos=Pose(
                x=roomba_start_x,
                y=roomba_start_y,
                theta=0.
            ), 
            dx=self.config.linear_speed/FPS,
            dtheta=self.config.rotational_speed/FPS,
            use_wraparound=self.config.wraparound
        )
        self.terminated = False
        # TODO: Make this a config param
        roomba_buffer=50
        free_space = (
            (roomba_start_x - roomba_buffer, roomba_start_y - roomba_buffer),
            (roomba_start_x + roomba_buffer, roomba_start_y + roomba_buffer),
        )

        if self.config.hardcode_particle_map:
            self._particles = HARDCODED_MAP
        else:
            self._particles = ParticleMap(
                self.config.n_particles,
                self.config.viewport_width,
                self.config.viewport_height, 
                free_space=free_space,
                max_dist=self.config.particle_speed,
                collision_dist=self.config.collision_dist
            )
        self._sensor = Sensor(SENSOR_DETECTION_THRESHOLD, SENSOR_ANGLES)
        self._i = 0
        self._bounds = (VIEWPORT_W, VIEWPORT_H)
        self._last_obs = self.measure()
        self._initial_distance = self._last_obs[0]
        return self._last_obs
        
    def __init__(self, roomba_env_config=None, render_mode="rgb_array", max_episode_steps=1000) -> None:
        super().__init__()
        if roomba_env_config is None:
            roomba_env_config = RoombaEnvConfig()
        self.config = roomba_env_config
        self.action_space = spaces.Discrete(4)
        if self.config.observation_space == SensorType.ULTRASONIC:
            # Observation space: (x, y, theta,) of goal
            low = [0, 0]
            high = [self.config.viewport_width**2+self.config.viewport_height**2, math.pi]
            # Observation space: (s1-s8) of sensor output
            low += [0.0 for i in range(N_SENSORS)]
            high += [100.0 for i in range(N_SENSORS)]
            low = np.array(low, dtype=np.float32)
            high = np.array(high, dtype=np.float32)
            self.observation_space = spaces.Box(low, high)
        elif self.config.observation_space == SensorType.VISUAL:
            self.observation_space = spaces.Box(
                    low=0, high=255, shape=(
                        self.config.viewport_width, self.config.viewport_height, 3
                    ), dtype=np.uint8
            )
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self._init_states()

    # TODO:Move to sensor
    def measure_ultrasonic(self):
        if self.config.observation_space == SensorType.VISUAL:
            raise Exception("Can't call this with visual observaton")
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        sensor_output = tuple(s*100 for s in sensor_output)
        return sensor_output

    def measure_distance(self):
        roomba_x, roomba_y, roomba_theta = self._roomba.pose
        goal_x, goal_y = self.goal
        dx = goal_x - roomba_x
        dy = goal_y - roomba_y
        # Calculate angle of goal to roomba

        goal_theta = math.atan2(dy, dx)
        diff = compute_angle_diff(goal_theta, roomba_theta)
        return (
                math.sqrt((goal_x - roomba_x)**2 + (goal_y - roomba_y)**2),
                diff, 
        ) 


    def measure(self):
        if self.config.observation_space == SensorType.VISUAL:
            return self.render(mode="rgb_array").transpose((1,0,2))
        else:
            obs = self.measure_distance()
            obs += self.measure_ultrasonic()
            return np.array(obs, dtype=np.float32)

    def calculate_reward_ultrasonic(self, obs):
        d_theta = abs(obs[1])
        distance = obs[0]
        last_distance = self._last_obs[0]
        # TODO: Make this a parameter
        OPPOSITE_THRESHOLD = math.pi/2
        CLOSE_THRESHOLD = math.pi/8
        # Three-tier : -1 if opposite direction, 1 if very close
        THETA_WEIGHT = 0 #1 if self._particles.n_particles == 0 else 0.01
        DISTANCE_THRESHOLD = 100
        DISTANCE_WEIGHT = 1 # DISTANCE_THRESHOLD / max(distance, 0.01)
        SENSOR_MAX = 100
        if d_theta > OPPOSITE_THRESHOLD:
            theta_reward = -5*THETA_WEIGHT
        elif d_theta < CLOSE_THRESHOLD:
            theta_reward = 2*THETA_WEIGHT
        else:
            theta_reward = 0
        # Maybe a front sensor punish?
        sensor_mid_dist = obs[2]
        sensor_punish = 0.01*(sensor_mid_dist - SENSOR_MAX)/SENSOR_MAX
        return (
                theta_reward + 
                DISTANCE_WEIGHT*(last_distance-distance)+
                sensor_punish
                - self.config.fuel_cost
        )

    def calculate_reward_general(self, distance):
        # You reached the goal!
        if distance < self.config.collision_dist:
            self.terminated = True
            return 1000
        # Punish going out of bounds
        x,y,theta = self._roomba.pose 
        if x > self.config.viewport_width or y > self.config.viewport_height or \
                x < 0 or y < 0:
            self.terminated = True
            return -500
        # You hit the particle :(
        if self._particles.detect_collision(self._roomba.pose):
            reward = -500
            self.terminated = True
            return reward
        if self.terminated:
            return 1000*((self._initial_distance - distance)/self._initial_distance)**2
        return 0

    def step(self, action):
        self._rendered_current = False
        # Reward is based on distance
        obs = self.measure()
        if self.terminated:
            return obs, 0, self.terminated, {}
        self._roomba.move(action, self._bounds)
        self._particles.move()
        distance=0
        if self.config.observation_space == SensorType.ULTRASONIC:
            reward = self.calculate_reward_ultrasonic(obs)
            distance = obs[0]
            self._last_obs = obs
        elif self.config.observation_space == SensorType.VISUAL:
            reward=0
            dist_array=self.measure_distance()
            distance = dist_array[0]
        reward += self.calculate_reward_general(distance)
        # TODO: consolidate rewards
        if self._i >= self._max_episode_steps:
            self.terminated = True
        # print(f"Reward: {reward}")
        # punish moving backwards
        if action == 2:
            reward -= 2
        self._i += 1
        return obs, reward, self.terminated, {}

    def reset(self, seed=0):
        return self._init_states(0)


    def render(self, mode=None):
        render_mode = mode if mode else self.render_mode
        # render_mode = self.render_mode

        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if not self._rendered_current:
            self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

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
            for sensor_level, angle_sensor in zip(sensor_output, self._sensor._angles):
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
            if self.config.visible_particles:
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
        
        self._rendered_current=True
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
    config = RoombaEnvConfig()
    config.n_particles = 1000
    config.particle_speed = 5
    config.observation_space = SensorType.VISUAL

    # Test the new env
    env_a_to_b = RoombaEnvAToB(roomba_env_config=config, render_mode="rgb_array")
    video_recorder_a_to_b = VideoRecorder(env_a_to_b, enabled=True, path='a_to_b.mp4')
    for i in range(4):
        for j in range(100):
            video_recorder_a_to_b.capture_frame()
            state, reward, terminated, _ = env_a_to_b.step(i)
            print(state)
            print(reward)
            print("===========")
        if terminated:
            break
    print(video_recorder_a_to_b.path)
    video_recorder_a_to_b.close()

