import json
from pathlib import Path
from ambf6dpose.DataCollection.BOPSaver.BopReader import BopReader
from ambf6dpose.DataCollection.BOPSaver.BopSaver import JsonFileManager


def main():
    root_path2 = "/home/juan1995/research_juan/accelnet_grant/BenchmarkFor6dObjectPose/BOP_datasets/ambf_suturing"
    reader = BopReader(
        root=Path(root_path2),
        scene_id_list=[],
        dataset_split="test",
        dataset_split_type="",
    )

    json_file = JsonFileManager("test.json", store_data_as="list")
    data = []
    with json_file:
        for i in range(len(reader)):
            scene_id, img_name = reader.get_metadata(i)
            img_id = int(img_name[:-4])
            data.append(
                {"im_id": int(img_id), "inst_count": 1, "obj_id": 1, "scene_id": int(scene_id)}
            )
        json_file.save_json(data)


if __name__ == "__main__":
    main()
