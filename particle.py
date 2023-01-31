from typing import NamedTuple
import random
import math

class Pose(NamedTuple):
    x: float
    y: float
    theta: float

            
class BaseParticle():
    def __init__(self, initial_pos: Pose):
        self.pose = initial_pos

    def _move_distance(self, dist, dtheta):
        prev_pos = self.pose
        theta = prev_pos.theta
        self.pose = Pose(
            x=prev_pos.x + dist * math.cos(theta),
            y=prev_pos.y + dist * math.sin(theta),
            theta=(theta + dtheta) % (2 * math.pi)
        )
        return self.pose

    def wraparound(self, bounds):
        if len(bounds) != 2:
            # TODO : specific error type for bad args?
            raise Exception("Error: Define bounds as (xmax, ymax) or ((xmin, ymin), (xmax, ymax))")
        if type(bounds[0]) == int:
            minx = miny = 0
            maxx = bounds[0]
            maxy = bounds[1]
        else:
            minx, miny = bounds[0]
            maxx, max_y = bounds[1]
        x_adjusted = self.pose.x - minx 
        y_adjusted = self.pose.y - miny 
        maxx_adjusted = maxx - minx
        maxy_adjusted = maxy - miny
        x_adjusted = x_adjusted % maxx_adjusted
        y_adjusted = y_adjusted % maxy_adjusted
        self.pose = Pose(
                x_adjusted + minx,
                y_adjusted + miny,
                self.pose.theta
        )
        return self.pose


class RandomParticle(BaseParticle):
    def __init__(self, pos: Pose, speed: float):
        super().__init__(pos)
        self._rnd = random.Random(int(pos.x) + 1024*int(pos.y))
        self._speed = speed # Speed per step

    def move(self, bounds):
        dist = self._speed 
        dtheta = self._rnd.random() * 2 * math.pi
        self._move_distance(dist, dtheta)
        return self.wraparound(bounds)


class ParticleMap():
    def __init__(self, n_particles, x_bound, y_bound, max_dist=2, collision_dist=4):
        self._particles = []
        for i in range(n_particles):
            start_pos = Pose(
                x=random.random() * x_bound, 
                y=random.random() * y_bound,
                theta=0
            )   
            self._particles.append(
                RandomParticle(start_pos, max_dist*random.random())
            )
        self._collision_dist = collision_dist
        self._bounds = (x_bound, y_bound)

    def move(self):
        for p in self._particles: 
            p.move(self._bounds)

    def detect_collision(self, pose):
        for particle in self._particles:
            dist = (particle.pose.x - pose.x)**2 + (particle.pose.y - pose.y)**2
            if dist < self._collision_dist ** 2:
                return True
        return False

    def get_readings(self, pose):
        readings = []
        for particle in self._particles:
            p = particle.pose
            angle = math.atan2(p.y - pose.y, p.x - pose.x)
            dist = (p.x - pose.x)**2 + (p.y - pose.y)**2
            readings.append((angle, math.sqrt(dist)))
        return readings

    @property
    def particles(self):
        return [p.pose for p in self._particles]

if __name__ == "__main__":
    p = RandomParticle(Pose(x=1, y=2, theta=0), speed=1)
    orig_pose = p.pose
    p.move((1000, 1000)) 
    print(p.pose)
    print((p.pose.x-orig_pose.x)**2 + (p.pose.y-orig_pose.y)**2)
    
    pmap = ParticleMap(5, 5, 10)
    pmap.move()
    print([r.pose for r in pmap._particles])

    p = BaseParticle(Pose(x=5, y=8, theta=0))
    p._move_distance(10, -math.pi)
    p.wraparound([10, 10])
    print("Should be [5, 8, math.pi]")
    print(p.pose)
    p._move_distance(10, math.pi/2)
    p.wraparound([10, 10])
    print("Should be [5, 8, 3*math.pi/2]")
    print(p.pose)
    p._move_distance(10, math.pi)
    p.wraparound([10, 10])
    print("Should be [5, 8, math.pi/2]")
    print(p.pose)
    p._move_distance(10, -math.pi/2)
    p.wraparound([10, 10])
    print("Should be [5, 8, 0]")
    print(p.pose)

