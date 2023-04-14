
import math

class Sensor:
    # TODO: Get rid of these magick numbers
    def __init__(self, detection_threshold, angles, tol=0.25):
        self.detection_threshold = detection_threshold
        self._tol = tol
        self._angles = angles
        # self._angle_delta_left = math.pi / 4
        # self._angle_delta_center = 0
        # self._angle_delta_right = -math.pi / 4

    def sense(self, roomba, particles):
        readings = particles.get_readings(roomba.pose)
        theta = roomba.pose.theta
        min_sensor_distances = [float('+inf') for i in self._angles]
        for angle, distance in readings:
            for i, sensor_angle in enumerate(self._angles):
                if abs(
                    angle - (theta + sensor_angle)
                ) < self._tol and distance < min_sensor_distances[i]:
                    min_sensor_distances[i] = distance
        return tuple(
                min(distance, self.detection_threshold) / self.detection_threshold
                for distance in min_sensor_distances
        )
            

