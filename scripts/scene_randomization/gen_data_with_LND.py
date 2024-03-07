from pathlib import Path
import random
import sys
import time
from typing import List

from click import Tuple
import click
from ambf6dpose.DataCollection.BOPSaver.BopSaver import BopSampleSaver
from ambf6dpose.DataCollection.RosClients import SyncRosInterface
from ambf6dpose.DataCollection.SimulatorDataProcessor import SimulatorDataProcessor
from ambf6dpose.SimulationObjs.psm_lnd.PSM_LND import LND
from dataclasses import dataclass, field
from ambf_client import Client
from PyKDL import Frame


@dataclass
class RpyTrajectoryGenerator:
    """
    Roll, pitch, yaw trajectory generator
    """

    initial_rpy: List[float]
    trajectory: List[float] = field(init=False, default_factory=list)
    step_size: float = 0.017  # 0.017 rad approx 1 deg
    total_steps: int = 5

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


@dataclass
class PSM1Traj(RpyTrajectoryGenerator):

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


@dataclass
class PSM2Traj(RpyTrajectoryGenerator):

    def __post_init__(self):
        self.generate_rpy_trajectory()

    def generate_rpy_trajectory(self) -> None:
        for i in range(0, self.total_steps):
            for j in range(0, self.total_steps):
                for k in range(0, self.total_steps):
                    self.trajectory.append(
                        [
                            self.initial_rpy[0] + j * self.step_size,
                            self.initial_rpy[1] + i * self.step_size,
                            self.initial_rpy[2] + k * self.step_size,
                        ]
                    )


@dataclass
class InstrumentPoseRandomizer:
    lnd_handle: LND
    initial_rpy: List[float] = field(init=False)

    def __post_init__(self):

        self.initial_rpy_psm2 = self.lnd_handle.base.get_pose()[3:]
        self.lnd_handle.set_jaw_angle(0.5)
        self.lnd_handle.servo_jp([0, 0, 0])

    def update(self, new_rpy: List[float]) -> None:
        self.lnd_handle.base.set_rpy(new_rpy[0], new_rpy[1], new_rpy[2])
        time.sleep(0.2)
        self.lnd_handle.servo_jp(self.gen_random_joint_angles())
        self.lnd_handle.set_jaw_angle(self.generate_random_jaw_angle())

    def reset(self):
        self.lnd_handle.set_jaw_angle(0.5)
        self.lnd_handle.servo_jp([0, 0, 0])

    @staticmethod
    def gen_random_joint_angles() -> List[float]:
        return [
            random.uniform(-1.3, 1.3),
            random.uniform(-0.7, 0.7),
            random.uniform(-0.7, 0.7),
        ]

    @staticmethod
    def generate_random_jaw_angle() -> float:
        return random.uniform(0.0, 0.7)


def init_client() -> Client:
    client = Client("LND_Simulation")
    client.connect()
    time.sleep(0.5)

    world_handle = client.get_world_handle()
    world_handle.reset_bodies()
    time.sleep(0.5)

    return client


@click.command()
@click.option("--total_steps", default=3, help="Total number of steps")
@click.option("--step_size", default=0.017, help="Step size")
def replay(total_steps, step_size):

    client = init_client()

    psm1_lnd = LND("/new_psm1/", client)
    psm2_lnd = LND("/new_psm2/", client)

    initial_rpy_psm1 = psm1_lnd.base.get_pose()[3:]
    initial_rpy_psm2 = psm2_lnd.base.get_pose()[3:]

    psm1_trajectory = PSM1Traj(
        initial_rpy_psm1, total_steps=total_steps, step_size=step_size
    )
    psm2_trajectory = PSM2Traj(
        initial_rpy_psm2, total_steps=total_steps, step_size=step_size
    )

    psm1_randomizer = InstrumentPoseRandomizer(psm1_lnd)
    psm2_randomizer = InstrumentPoseRandomizer(psm2_lnd)

    for t1, t2 in zip(psm1_trajectory, psm2_trajectory):
        psm1_randomizer.update(t1)
        psm2_randomizer.update(t2)

        time.sleep(1.4)

    psm1_randomizer.reset()
    psm2_randomizer.reset()

    time.sleep(0.5)


def wait_for_data(client: SyncRosInterface):
    try:
        client.wait_for_data(12)
    except TimeoutError:
        print(
            "ERROR: Timeout exception triggered. ROS message filter did not receive any data.",
            file=sys.stderr,
        )
        sys.exit(1)


@click.command()
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory",
)
@click.option("--scene_id", help="id given to current collection",required=True, type=int)
@click.option("--total_steps", default=4, help="Total number of steps")
@click.option("--step_size", default=0.012, help="Step size")
def replay_and_record(output_dir, scene_id, total_steps, step_size):

    output_dir = Path(output_dir)
    saver: BopSampleSaver = BopSampleSaver(output_dir, scene_id=scene_id)
    ros_client = SyncRosInterface()
    samples_generator = SimulatorDataProcessor(ros_client)

    wait_for_data(ros_client)

    client = init_client()
    psm1_lnd = LND("/new_psm1/", client)
    psm2_lnd = LND("/new_psm2/", client)

    initial_rpy_psm1 = psm1_lnd.base.get_pose()[3:]
    initial_rpy_psm2 = psm2_lnd.base.get_pose()[3:]

    psm1_trajectory = PSM1Traj(
        initial_rpy_psm1, total_steps=total_steps, step_size=step_size
    )
    psm2_trajectory = PSM2Traj(
        initial_rpy_psm2, total_steps=total_steps, step_size=step_size
    )

    psm1_randomizer = InstrumentPoseRandomizer(psm1_lnd)
    psm2_randomizer = InstrumentPoseRandomizer(psm2_lnd)

    count = 0
    with saver:
        for t1, t2 in zip(psm1_trajectory, psm2_trajectory):
            psm1_randomizer.update(t1)
            psm2_randomizer.update(t2)

            time.sleep(1.4)
            wait_for_data(samples_generator.simulation_client)
            data_sample = samples_generator.generate_dataset_sample()
            saver.save_sample(data_sample)
            print(f"Collected sample: {count}")
            count += 1

    psm1_randomizer.reset()
    psm2_randomizer.reset()

    time.sleep(0.5)


@click.group(help="Motion generation scripts")
def main():
    pass


if __name__ == "__main__":
    main.add_command(replay_and_record)
    main.add_command(replay)
    main()
