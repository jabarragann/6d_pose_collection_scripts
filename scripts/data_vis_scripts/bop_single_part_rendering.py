from pathlib import Path
from typing import List
import cv2
import numpy as np
from ambf6dpose.DataCollection.BOPSaver.BopReader import BopDatasetReader
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataVisualization.bop_vis_utils import (
    rendering_gt_single_obj
)


def main():
    file_path = Path(__file__).resolve().parent
    dataset_path = file_path / "../../SampleData/BOP/needle_gripper_dataset_V0.0.2"
    dataset_path = dataset_path.resolve()
    needle_model_path = file_path / "../../SampleData/Models/Needle.ply"
    needle_model_path = needle_model_path.resolve()
    toolpitchlink_model_path = file_path / "../../SampleData/Models/NewPSMToolPitchLink.ply"
    toolpitchlink_model_path = toolpitchlink_model_path.resolve()

    assert dataset_path.exists(), f"Path {dataset_path} does not exist"
    assert needle_model_path.exists(), f"Path {needle_model_path} does not exist"
    assert (
        toolpitchlink_model_path.exists()
    ), f"Path {toolpitchlink_model_path} does not exist"

    reader = BopDatasetReader(
        root=Path(dataset_path),
        scene_id_list=[],
        dataset_split="test",
        dataset_split_type="",
    )

    print(f"Dataset size: {len(reader)}")

    for idx in range(0, len(reader)):
        sample: DatasetSample = reader[idx]
        scene_id, img_name = reader.get_metadata(idx)

        img_with_gt, ren_out = rendering_gt_single_obj(toolpitchlink_model_path, sample)
        final = np.hstack((img_with_gt, ren_out))
        cv2.imshow("image", final)
        print(f"{scene_id}-{img_name}")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
