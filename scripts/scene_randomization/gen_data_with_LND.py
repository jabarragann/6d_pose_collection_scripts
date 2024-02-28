import random
import time
from typing import List
from ambf6dpose.SimulationObjs.psm_lnd.PSM_LND import LND
from dataclasses import dataclass, field
from PyKDL import Frame


@dataclass
class RpyTrajectoryGenerator:
    """
    Roll, pitch, yaw trajectory generator
    """

    initial_rpy: List[float]
    trajectory: List[float] = field(init=False)
    step_size: float = 0.017  # 0.017 rad approx 1 deg
    total_steps: int = 5

    def __post_init__(self):
        self.generate_rpy_trajectory()

    def generate_rpy_trajectory(self) -> None:
        self.trajectory = []

        for i in range(0, self.total_steps):
            for j in range(0, self.total_steps):
                for k in range(0, self.total_steps):
                    self.trajectory.append(
                        [
                            self.initial_rpy[0] - i * self.step_size,
                            self.initial_rpy[1] - j * self.step_size,
                            self.initial_rpy[2] - k * self.step_size,
                        ]
                    )

    def __iter__(self):
        self.__internal_idx = 0
        return self

    def __next__(self):
        if self.__internal_idx < len(self.trajectory):
            sample = self.trajectory[self.__internal_idx]
            self.__internal_idx += 1
            return sample
        else:
            raise StopIteration


def gen_random_joint_angles() -> List[float]:
    return [
        random.uniform(-1.3, 1.3),
        random.uniform(-0.7, 0.7),
        random.uniform(-0.7, 0.7),
    ]


def generate_random_jaw_angle() -> float:
    return random.uniform(0.0, 0.7)


def replay_trajectory():
    lnd = LND("test")
    initial_rpy = lnd.base.get_pose()[3:]

    trajectory = RpyTrajectoryGenerator(initial_rpy, total_steps=3)

    lnd.set_jaw_angle(0.5)
    lnd.servo_jp([0, 0, 0])

    for t in trajectory:
        jaw_angle = generate_random_jaw_angle()
        print(f"{t}")
        print(f"{jaw_angle = }")
        lnd.base.set_rpy(t[0], t[1], t[2])
        time.sleep(0.3)
        lnd.servo_jp(gen_random_joint_angles())
        time.sleep(0.3)
        lnd.set_jaw_angle(jaw_angle)

        time.sleep(1.2)

def generate_data_with_LND():
    pass

if __name__ == "__main__":
    replay_trajectory()
