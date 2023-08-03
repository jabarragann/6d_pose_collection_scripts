from pathlib import Path
import click
import cv2
from ambf6dpose import DatasetReader, SampleSaver


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
def generate_dataset(path: str):
    """Generate test images by project needle 3d point to the image plane. This script can be used
    to visualy inspect the correctness of the intrinsic and extrinsic matrices
    """
    path = Path(path).resolve()
    dataset = DatasetReader(path)

    blended_path = path / "blended_img"
    blended_path.mkdir(exist_ok=True)

    for idx, sample in enumerate(dataset):
        fmt_idx = SampleSaver.fmt_step(idx)
        sample.generate_blended_img()
        cv2.imwrite(str(blended_path / f"{fmt_idx}.png"), sample.blended_img)
        print(f"Sample {fmt_idx}")
        print(sample.extrinsic_matrix)


if __name__ == "__main__":
    generate_dataset()
