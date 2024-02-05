from pathlib import Path
from typing import List
import click
from ambf6dpose.DataCollection.BOPSaver.BopReader import BopDatasetReader
from ambf6dpose.DataCollection.BOPSaver.BopSaver import (
    BopSampleSaver,
    JsonSaver,
    get_folder_names,
)
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.ReaderSaverUtils import ImgDirs
import cv2
from contextlib import ExitStack
import shutil
from tqdm import tqdm

# root_path = "/home/juan1995/research_juan/accelnet_grant/6d_pose_dataset_collection/test_ds_bop"
# root_path2 = "/home/juan1995/research_juan/accelnet_grant/BenchmarkFor6dObjectPose/BOP_datasets/ambf_suturing"


class SaversManager:
    def __init__(self) -> None:
        self.exit_stack = ExitStack()

    def create_savers(self, root: Path, scene_id_list: List[str]):
        self.root = root
        self.scene_id_list = scene_id_list

        return self

    def __enter__(self):
        self.savers_dict = {}
        for id in self.scene_id_list:
            assert isinstance(id, str), f"scene_id_list must be a list of strings"
            saver = JsonSaver(
                self.root / id,
                save_every=1,
                scene_camera_name="scene_camera_corrected.json",
                scene_gt_name="scene_gt_corrected.json",
                safe_save=False,
            )
            self.exit_stack.enter_context(saver)
            self.savers_dict[id] = saver
        return self.savers_dict

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.close()


def filter_imgs(root: Path, reader: BopDatasetReader):
    img_folders = get_folder_names()
    removed_images = root.parent / (root.name + "_removed_images")
    saver_manager = SaversManager()

    with saver_manager.create_savers(root, reader.scene_id_list) as savers_dict:
        for idx, sample in tqdm(
            enumerate(reader), total=len(reader), desc="Filtering images"
        ):
            scene_id, img_name = reader.get_metadata(idx)
            img_id = int(img_name[:-4])

            img_pt = sample.project_needle_points().squeeze()
            test_x = (img_pt[:, 0] > 0) & (img_pt[:, 0] < 640)
            test_y = (img_pt[:, 1] > 0) & (img_pt[:, 1] < 480)
            valid_points = test_x & test_y
            valid_points = valid_points.sum()

            if valid_points < 5:
                print(
                    f"{idx} scene:{scene_id}-{img_name} has {valid_points} valid points"
                )

                # remove images
                move_path = removed_images / scene_id
                for img_type in ImgDirs:
                    src_p = root / scene_id / img_folders[img_type] / img_name
                    dst_p = move_path / img_folders[img_type] / img_name
                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                    if src_p.exists():
                        shutil.move(src_p, dst_p)

            else:
                # Save valid samples
                savers_dict[scene_id].save_sample(img_id, sample)

    # cp original gt files to removed
    for scene_id in reader.scene_id_list:
        for name in ["scene_gt.json", "scene_camera.json"]:
            src_cp = root / scene_id / name
            dst_cp = removed_images / scene_id / name
            dst_cp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_cp, dst_cp)

    # rename original gt files to  old
    for scene_id in reader.scene_id_list:
        for name in ["scene_gt.json", "scene_camera.json"]:
            src_cp = root / scene_id / name
            dst_cp = root / scene_id / name
            dst_cp = dst_cp.parent / (dst_cp.with_suffix("").name + "_old.json")
            shutil.move(src_cp, dst_cp)

    # rename new gt files
    for scene_id in reader.scene_id_list:
        src_cp = root / scene_id / "scene_gt_corrected.json"
        dst_cp = root / scene_id / "scene_gt.json"
        shutil.move(src_cp, dst_cp)

        src_cp = root / scene_id / "scene_camera_corrected.json"
        dst_cp = root / scene_id / "scene_camera.json"
        shutil.move(src_cp, dst_cp)

    # print(img_pt)
    # print(test_x)
    # print(test_y)
    # idx = 0
    # sample: DatasetSample = reader[idx]
    # scene_id, img_name = reader.get_metadata(idx)
    # sample.generate_gt_vis()
    # cv2.imshow(f"{scene_id}-{img_name}", sample.gt_vis_img)
    # cv2.waitKey(0)


@click.command()
@click.option(
    "--root_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
)
@click.option(
    "-s", "--dataset_split", help="train or test", type=click.Choice(["train", "test"])
)
@click.option(
    "-t",
    "--dataset_split_type",
    default=None,
    help="ds split type. see bop toolkit data format.",
)
def filter_img_without_needle(
    root_path: Path, dataset_split: str, dataset_split_type: str
):
    reader = BopDatasetReader(
        root=Path(root_path),
        scene_id_list=[],
        dataset_split=dataset_split,
        dataset_split_type=dataset_split_type,
    )

    print("Loading:")
    reader.print_ds_info()

    filter_imgs(reader.root, reader)


if __name__ == "__main__":
    filter_img_without_needle()
