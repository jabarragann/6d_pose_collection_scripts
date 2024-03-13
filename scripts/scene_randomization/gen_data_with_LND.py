from pathlib import Path
import random
import sys
import time
from typing import List

from click import Tuple
import click
from click_params import IntListParamType
import numpy as np
from ambf6dpose.DataCollection.BOPSaver.BopSaver import BopSampleSaver
from ambf6dpose.DataCollection.RosClients import SyncRosInterface
from ambf6dpose.DataCollection.SimulatorDataProcessor import SimulatorDataProcessor
from ambf6dpose.SimulationObjs.psm_lnd.PSM_LND import LND
from dataclasses import dataclass, field
from ambf_client import Client
from PyKDL import Frame
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager


def get_camera_positions() -> List[List[float]]:
    ecm_list = []
    ecm_list.append([0.0, 0.0, 0.0, 0.0])  # 0
    ecm_list.append([0.0, 0.05, -0.00, 0.0])  # 1
    ecm_list.append([0.0, 0.05, -0.00, 0.4])  # 2
    ecm_list.append([0.0, 0.05, -0.00, -0.4])  # 3
    ecm_list.append([0.2, 0.05, -0.00, 0.0])  # 4
    ecm_list.append([-0.2, 0.05, -0.00, 0.0])  # 5
    ecm_list.append([0.0, 0.15, -0.00, 0.0])  # 6
    ecm_list.append([0.0, -0.05, -0.00, 0.0])  # 7
    ecm_list.append([0.0, 0.05, -0.005, 0.0])  # 8
    ecm_list.append([0.0, 0.05, -0.005, 0.0])  # 9
    ecm_list.append([0.1, 0.05, -0.005, 0.2])  # 10
    ecm_list.append([-0.1, 0.05, -0.005, -0.2])  # 11
    ecm_list.append([0.1, 0.10, 0.001, 0.0])  # 12
    ecm_list.append([-0.1, 0.0, 0.001, 0.0])  # 13
    ecm_list.append([0.0, 0.10, -0.005, 0.3])  # 14
    ecm_list.append([0.0, 0.0, 0.001, 0.3])  # 15
    ecm_list.append([0.1, 0.10, -0.006, 0.1])  # 16
    ecm_list.append([-0.1, 0.0, -0.006, -0.1])  # 17
    ecm_list.append([0.1, 0.10, -0.003, -0.1])  # 18
    ecm_list.append([-0.1, 0.0, 0.00, 0.1])  # 19

    # Add random noise to ECM joint positions
    for i in range(len(ecm_list)):
        ecm_list[i][0] += 0.06 * (2 * np.random.rand() - 1)
        ecm_list[i][1] += 0.06 * (2 * np.random.rand() - 1)
        ecm_list[i][2] += 0.003 * (2 * np.random.rand() - 1)
        ecm_list[i][3] += 0.08 * (2 * np.random.rand() - 1)

    return ecm_list


@dataclass
class PoseTrajectoryGenerator:
    """
    Roll, pitch, yaw trajectory generator
    """

    initial_pos: List[float]
    initial_rpy: List[float]
    trajectory: List[float] = field(init=False, default_factory=list)
    pos_step_size: float = 0.015  # 5mm
    rot_step_size: float = 0.017  # 0.017 rad approx 1 deg
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

    def gen_random_triplet(self, weight: float) -> List[float]:
        return [
            weight * random.gauss(mu=0.0, sigma=1.0),
            weight * random.gauss(mu=0.0, sigma=1.0),
            weight * random.gauss(mu=0.0, sigma=1.0),
        ]


@dataclass
class PSM1Traj(PoseTrajectoryGenerator):

    def __post_init__(self):
        self.generate_rpy_trajectory()

    def generate_rpy_trajectory(self) -> None:
        self.trajectory = []

        for i in range(0, self.total_steps):
            for j in range(0, self.total_steps):
                for k in range(0, self.total_steps):
                    random_pos_offset = self.gen_random_triplet(self.pos_step_size)
                    self.trajectory.append(
                        [
                            self.initial_pos[0] - random_pos_offset[0],
                            self.initial_pos[1] - random_pos_offset[1],
                            self.initial_pos[2] - random_pos_offset[2],
                            self.initial_rpy[0] - i * self.rot_step_size,
                            self.initial_rpy[1] - j * self.rot_step_size,
                            self.initial_rpy[2] - k * self.rot_step_size,
                        ]
                    )


@dataclass
class PSM2Traj(PoseTrajectoryGenerator):

    def __post_init__(self):
        self.generate_rpy_trajectory()

    def generate_rpy_trajectory(self) -> None:
        for i in range(0, self.total_steps):
            for j in range(0, self.total_steps):
                for k in range(0, self.total_steps):

                    random_pos_offset = self.gen_random_triplet(self.pos_step_size)
                    self.trajectory.append(
                        [
                            self.initial_pos[0] - random_pos_offset[0],
                            self.initial_pos[1] - random_pos_offset[1],
                            self.initial_pos[2] - random_pos_offset[2],
                            self.initial_rpy[0] + j * self.rot_step_size,
                            self.initial_rpy[1] + i * self.rot_step_size,
                            self.initial_rpy[2] + k * self.rot_step_size,
                        ]
                    )


@dataclass
class InstrumentJointRandomizer:
    lnd_handle: LND

    def __post_init__(self):

        self.initial_rpy_psm2 = self.lnd_handle.base.get_pose()[3:]
        self.lnd_handle.set_jaw_angle(0.5)
        self.lnd_handle.servo_jp([0, 0, 0])

    def update(self, new_pos: List[float], new_rpy: List[float]) -> None:
        self.lnd_handle.base.set_pos(new_pos[0], new_pos[1], new_pos[2])
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


class DummySaver:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def save_sample(self, *args):
        pass


def init_client() -> Client:
    client = Client("LND_Simulation")
    client.connect()
    time.sleep(0.5)

    world_handle = client.get_world_handle()
    world_handle.reset_bodies()
    time.sleep(0.5)

    return client


def wait_for_data(client: SyncRosInterface):
    try:
        client.wait_for_data(12)
    except TimeoutError:
        print(
            "ERROR: Timeout exception triggered. ROS message filter did not receive any data.",
            file=sys.stderr,
        )
        sys.exit(1)


def extract_wanted_ecm_positions(wanted_positions: List[int]) -> List[List[float]]:

    ecm_full_list = get_camera_positions()

    if len(wanted_positions) == 0:
        wanted_positions = list(range(0, len(ecm_full_list)))

    assert all(
        [i >= 0 for i in wanted_positions]
    ), "ECM positions must be positive integers"
    assert all(
        [i < len(ecm_full_list) for i in wanted_positions]
    ), f"Only {len(ecm_full_list)} ECM positions are available"

    ecm_jp_list = [ecm_full_list[i] for i in wanted_positions]

    return ecm_jp_list


@dataclass
class ExperimentManager:
    total_steps: int
    step_size: float

    def __post_init__(self):

        self.client = init_client()
        self.sim_manager = SimulationManager("record_test")
        self.cam_frame_handle = ECM(self.sim_manager, "CameraFrame")

        self.init_psm_randomizers()

    def init_psm_randomizers(self):
        self.psm1_lnd = LND("/new_psm1/", self.client)
        self.psm2_lnd = LND("/new_psm2/", self.client)

        initial_pos_psm1 = self.psm1_lnd.base.get_pose()[:3]
        initial_pos_psm2 = self.psm2_lnd.base.get_pose()[:3]
        initial_rpy_psm1 = self.psm1_lnd.base.get_pose()[3:]
        initial_rpy_psm2 = self.psm2_lnd.base.get_pose()[3:]

        self.psm1_trajectory = PSM1Traj(
            initial_pos_psm1,
            initial_rpy_psm1,
            total_steps=self.total_steps,
            rot_step_size=self.step_size,
        )
        self.psm2_trajectory = PSM2Traj(
            initial_pos_psm2,
            initial_rpy_psm2,
            total_steps=self.total_steps,
            rot_step_size=self.step_size,
        )

        self.psm1_randomizer = InstrumentJointRandomizer(self.psm1_lnd)
        self.psm2_randomizer = InstrumentJointRandomizer(self.psm2_lnd)

    def get_psm_trajectories(self):
        return self.psm1_trajectory, self.psm2_trajectory

    def reset_psm(self):
        self.psm1_randomizer.reset()
        self.psm2_randomizer.reset()


@click.command("replay-and-record", context_settings={"show_default": True})
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory",
)
@click.option("--scene_id", help="id given to current collection", type=int)
@click.option("--total_steps", default=2, help="Total number of steps")
@click.option("--step_size", default=0.012, help="Step size")
@click.option("--record", is_flag=True, default=False, help="Record data flag.")
@click.option(
    "--ecm_positions",
    default="",
    help="Index of ECM positions, specifid as a string of ints separated with a space, \
    e.g., '1 2 3'. Default value: all available positions",
    type=IntListParamType(" ", ignore_empty=True),
)
def replay_and_record(
    output_dir, scene_id, total_steps, step_size, record, ecm_positions
):

    if record:
        if output_dir is None:
            raise ValueError("Output directory must be provided if recording data")
        if scene_id is None:
            raise ValueError("Scene id must be provided if recording data")

        output_dir = Path(output_dir)
        saver: BopSampleSaver = BopSampleSaver(output_dir, scene_id=scene_id)
    else:
        saver = DummySaver()

    ros_client = SyncRosInterface()
    samples_generator = SimulatorDataProcessor(ros_client)
    wait_for_data(ros_client)

    manager = ExperimentManager(total_steps, step_size)
    camera_positions = extract_wanted_ecm_positions(ecm_positions)

    count = 0
    with saver:
        for camera_pos_idx, cam_jp in enumerate(camera_positions):
            manager.cam_frame_handle.servo_jp(cam_jp)
            psm1_traj, psm2_traj = manager.get_psm_trajectories()

            for t1, t2 in zip(psm1_traj, psm2_traj):
                t1_pos, t1_rpy = t1[:3], t1[3:]
                t2_pos, t2_rpy = t2[:3], t2[3:]
                manager.psm1_randomizer.update(t1_pos, t1_rpy)
                manager.psm2_randomizer.update(t2_pos, t2_rpy)

                time.sleep(1.4)
                wait_for_data(samples_generator.simulation_client)
                data_sample = samples_generator.generate_dataset_sample()
                saver.save_sample(data_sample)
                print(f"Collected sample: {count}")
                count += 1

    manager.reset_psm()

    time.sleep(0.5)


@click.group(help="Motion generation scripts")
def main():
    pass


if __name__ == "__main__":
    main.add_command(replay_and_record)
    main()
