from pathlib import Path
import click
from ambf6dpose import RawDataProcessor, SampleSaver
import rospy
import time
from ambf6dpose import AbstractSimulationClient, SyncRosInterface, RosInterface, AMBFClientWrapper


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--sample_time", default=0.5, help="Sample every n seconds")
def collect_data(path: str, sample_time: float) -> None:
    """6D pose data collection script.
    Instructions: (1) Run ambf simulation (2) run recorded motions (3) run collection script.
    """

    path = Path(path).resolve()

    client = SyncRosInterface()
    client.wait_until_first_sample()

    samples_generator = RawDataProcessor(client)
    saver = SampleSaver(root=path)

    last_time = time.time() + sample_time
    count = 0
    while not rospy.is_shutdown():
        if time.time() - last_time > sample_time:
            sample = samples_generator.generate_dataset_sample()
            saver.save_sample(sample)
            print(f" Sample: {count} Time: {time.time()-last_time:0.3f}")
            last_time = time.time()
            count += 1
    saver.close()


if __name__ == "__main__":
    collect_data()
