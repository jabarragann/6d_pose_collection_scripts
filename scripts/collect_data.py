from pathlib import Path
import sys
import click
from ambf6dpose import SimulatorDataProcessor
from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import YamlSampleSaver
import rospy
import time
from ambf6dpose import AbstractSimulationClient, SyncRosInterface
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver
from ambf6dpose.DataCollection.BOPSaver.BopSaver import BopSampleSaver
import signal


def signal_handler(sig, frame):
    print("\nClosing collection script")
    sys.exit(0)


def create_sample_saver(root: Path, type: str, scene_id: int) -> AbstractSaver:
    if type == "bop":
        return BopSampleSaver(root, scene_id=scene_id)
    elif type == "yaml":
        from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import YamlSampleSaver

        return YamlSampleSaver(root)
    else:
        raise ValueError(f"Unknown sampler saver {type}")


def wait_for_data(client: AbstractSimulationClient):
    try:
        client.wait_for_data()
    except TimeoutError:
        print(
            "ERROR: Timeout exception triggered. ROS message filter did not receive any data.",
            file=sys.stderr,
        )
        sys.exit(1)


def start_collection(
    samples_generator: SimulatorDataProcessor, saver: BopSampleSaver, sample_time: float
):
    last_time = time.time() + sample_time
    count = 0

    with saver:
        while not rospy.is_shutdown():
            if time.time() - last_time > sample_time:
                wait_for_data(samples_generator.simulation_client)  # can trigger timeout exception

                sample = samples_generator.generate_dataset_sample()
                saver.save_sample(sample)
                print(f" Sample: {count} Time from last sample: {time.time()-last_time:0.3f}")
                last_time = time.time()
                count += 1


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--scene_id", required=True, help="scene_id")
@click.option("--sample_time", default=0.5, help="Sample every n seconds")
def collect_data(path: str, scene_id: int, sample_time: float) -> None:
    """6D pose data collection script.
    Instructions: (1) Run ambf simulation (2) run recorded motions (3) run collection script.
    """

    # Setup
    signal.signal(signal.SIGINT, signal_handler)  # ctrl+c handler
    path = Path(path).resolve()
    scene_id = int(scene_id)
    client = SyncRosInterface()

    samples_generator = SimulatorDataProcessor(client)
    saver: BopSampleSaver = create_sample_saver(path, "bop", scene_id)

    start_collection(samples_generator, saver, sample_time)


if __name__ == "__main__":
    collect_data()
