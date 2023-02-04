import enum
import math
from particle import Pose, BaseParticle

class Action(enum.Enum):
  FORWARD=0
  LEFT=1
  BACKWARD=2
  RIGHT=3

class Roomba(BaseParticle):
    def __init__(self, pos: Pose, dx: float, dtheta: float) -> None:
        super().__init__(pos)
        self._dx = dx
        self._dtheta = dtheta

    def move(self, action: int, bounds: tuple):
        action_enum = Action(action)
        if action_enum == Action.FORWARD:
            self._move_distance(self._dx, 0)
        elif action_enum == Action.LEFT:
            self._move_distance(0, self._dtheta)
        elif action_enum == Action.BACKWARD:
            self._move_distance(-self._dx, 0)
        elif action_enum == Action.RIGHT:
            self._move_distance(0, -self._dtheta)
        self.wraparound(bounds)

    def pose(self):
      return self._pos
