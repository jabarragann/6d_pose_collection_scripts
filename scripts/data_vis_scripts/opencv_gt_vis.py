import copy
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from ambf6dpose import YamlDatasetReader
from surgical_robotics_challenge.units_conversion import SimToSI

from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataVisualization.opencv_vis_utils import load_mesh, opencv_gt_vis


def main():
    current_dir = Path(__file__).resolve().parent
    mesh_path = current_dir / "../../SampleData/Models/Needle.ply"
    ds_path = current_dir / "../../SampleData/YAML/needle_dataset"
    dataset = YamlDatasetReader(ds_path)

    vertices, mesh = load_mesh(mesh_path)

    available_samples = [0, 10, 20, 30, 40, 50, 60]
    for idx in available_samples:
        annotated_img = opencv_gt_vis(vertices, dataset[idx])
        cv2.imshow("img", annotated_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
