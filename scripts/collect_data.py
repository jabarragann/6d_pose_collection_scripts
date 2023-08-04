from pathlib import Path
import click
from ambf6dpose import SimulationInterface, SampleSaver
import rospy
import time
import copy

# Config file
## path
## segmentation camera


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--sample_time", default=0.5, help="Sample every n seconds")

def collect_data(path: str, sample_time: float) -> None:
    """6D pose data collection script.
    Instructions: (1) Run ambf simulation (2) run recorded motions (3) run collection script.

    """
    path = Path(path).resolve()
    sim_interface = SimulationInterface()
    saver = SampleSaver(root=path)

    # last_time = time.time_ns() + sample_time
    last_time = time.time_ns()

    count = 0
    while not rospy.is_shutdown():
        if (time.time_ns() - last_time)*1e-9 - sample_time > 1e-6:
            sample = sim_interface.generate_dataset_sample()
            sample_temp = copy.deepcopy(sample)
            saver.save_sample(sample_temp)
            print(f" Sample: {count} Time: {(time.time_ns()-last_time)*1e-9}")
            last_time = time.time_ns()
            count +=1

    saver.close()


if __name__ == "__main__":
    collect_data()
