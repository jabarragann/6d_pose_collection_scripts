from pathlib import Path
import click
from ambf6dpose import SimulatorDataProcessor
from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import SampleSaver
import rospy
import time
from ambf6dpose import AbstractSimulationClient, SyncRosInterface
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver

def create_sample_saver(root: Path, type:str)-> AbstractSaver: 
    if type == "bop":
        from ambf6dpose.DataCollection.BOPSaver.BopSaver import SampleSaver
        return  SampleSaver(root, scene_id=0)
    elif type == "yaml":
        from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import SampleSaver
        return SampleSaver(root)
    else:
        raise ValueError(f"Unknown sampler saver {type}")

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

    samples_generator = SimulatorDataProcessor(client)
    saver = create_sample_saver(path, "bop") 

    last_time = time.time() + sample_time
    count = 0

    with saver:
        while not rospy.is_shutdown():
            if time.time() - last_time > sample_time:
                sample = samples_generator.generate_dataset_sample()
                saver.save_sample(sample)
                print(f" Sample: {count} Time: {time.time()-last_time:0.3f}")
                last_time = time.time()
                count += 1


if __name__ == "__main__":
    collect_data()
