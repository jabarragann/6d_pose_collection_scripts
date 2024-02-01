from pathlib import Path
import click
import cv2
from ambf6dpose import DatasetReader, YamlSampleSaver


@click.command()
@click.option("--path", required=True, help="Path to save dataset")
def generate_dataset(path: str):
    """Generate test images by projecting needle 3d points to the image plane. This script can be used
    to visualy inspect the correctness of the intrinsic and extrinsic matrices
    """
    path = Path(path).resolve()
    dataset = DatasetReader(path)

    blended_path = path / "blended_img"
    blended_path.mkdir(exist_ok=True)

    for idx, sample in enumerate(dataset):
        fmt_idx = YamlSampleSaver.fmt_step(idx)
        sample.generate_gt_vis()
        cv2.imwrite(str(blended_path / f"{fmt_idx}.png"), sample.gt_vis_img)
        print(f"Sample {fmt_idx}")
        print(sample.needle_pose)


if __name__ == "__main__":
    generate_dataset()
