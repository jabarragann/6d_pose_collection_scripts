from pathlib import Path
import click
from ambf6dpose import SimulationInterface, SampleSaver
import rospy
import time

# Config file
## path
## segmentation camera


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--sample_time", default=1.5, help="Sample every n seconds")
def collect_data(path: str, sample_time: float) -> None:
    """6D pose data collection script.
    Instructions: (1) Run ambf simulation (2) run recorded motions (3) run collection script.

    """
    path = Path(path).resolve()
    sim_interface = SimulationInterface()
    saver = SampleSaver(root=path)

    last_time = time.time() + sample_time

    count = 0
    while not rospy.is_shutdown():
        if time.time() - last_time > sample_time:
            sample = sim_interface.generate_dataset_sample()
            saver.save_sample(sample)
            print(f" Sample: {count} Time: {time.time()-last_time:0.3f}")
            last_time = time.time()
            count +=1

    saver.close()


if __name__ == "__main__":
    collect_data()
