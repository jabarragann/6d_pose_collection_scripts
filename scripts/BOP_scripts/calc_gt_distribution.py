# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates distribution of GT poses."""
from dataclasses import dataclass, field
import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
import pandas as pd
import seaborn as sns
from matplotlib import pylab


# PARAMETERS.
################################################################################
p = {
    # See dataset_params.py for options.
    "dataset": "ambf_suturing",
    # Dataset split. Options: 'train', 'val', 'test'.
    "dataset_split": "test",
    # Dataset split type. None = default. See dataset_params.py for options.
    "dataset_split_type": None,
    # Folder containing the BOP datasets.
    "datasets_path": config.datasets_path,
}
################################################################################

params = {
    "figure.titlesize": "xx-large",
    "legend.fontsize": "x-large",
    "legend.title_fontsize": "x-large",
    "figure.figsize": (9, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

adjust_params = dict(top=0.88, bottom=0.18, left=0.125, right=0.9, hspace=0.2, wspace=0.25)


@dataclass
class GtDistCalculator:
    dp_split: dict
    dists: List = field(default_factory=list, init=False)
    azimuths: List = field(default_factory=list, init=False)
    elevs: List = field(default_factory=list, init=False)
    visib_fracts: List = field(default_factory=list, init=False)
    ims_count: int = field(default_factory=int, init=False)

    def __post_init__(self):
        self.scene_ids = self.dp_split["scene_ids"]

    def get_dp_split(self):
        return self.dp_split

    def calc_dist_for_scene_id(self, scene_gt, scene_gt_info):
        for im_id in scene_gt.keys():
            for gt_id, im_gt in enumerate(scene_gt[im_id]):
                # Object distance.
                dist = np.linalg.norm(im_gt["cam_t_m2c"])
                self.dists.append(dist)

                # Camera origin in the model coordinate system.
                cam_orig_m = -np.linalg.inv(im_gt["cam_R_m2c"]).dot(im_gt["cam_t_m2c"])

                # Azimuth from [0, 360].
                azimuth = math.atan2(cam_orig_m[1, 0], cam_orig_m[0, 0])
                if azimuth < 0:
                    azimuth += 2.0 * math.pi
                self.azimuths.append((180.0 / math.pi) * azimuth)

                # Elevation from [-90, 90].
                a = np.linalg.norm(cam_orig_m)
                b = np.linalg.norm([cam_orig_m[0, 0], cam_orig_m[1, 0], 0])
                elev = math.acos(b / a)
                if cam_orig_m[2, 0] < 0:
                    elev = -elev
                self.elevs.append((180.0 / math.pi) * elev)

                # Visibility fraction.
                self.visib_fracts.append(scene_gt_info[im_id][gt_id]["visib_fract"])

    def calc_gt_dist(self):
        for scene_id in self.scene_ids:
            misc.log(
                "Processing - dataset: {} ({}, {}), scene: {}".format(
                    p["dataset"], p["dataset_split"], p["dataset_split_type"], scene_id
                )
            )

            # Load GT poses.
            scene_gt = inout.load_scene_gt(dp_split["scene_gt_tpath"].format(scene_id=scene_id))

            # Load info about the GT poses.
            scene_gt_info = inout.load_json(
                dp_split["scene_gt_info_tpath"].format(scene_id=scene_id), keys_to_int=True
            )

            self.ims_count += len(scene_gt)

            self.calc_dist_for_scene_id(scene_gt, scene_gt_info)

    def print_stats(self):
        # Print stats.
        misc.log("Stats of the GT poses in dataset {} {}:".format(p["dataset"], p["dataset_split"]))
        misc.log("Number of images: " + str(self.ims_count))

        misc.log("Min dist: {}".format(np.min(self.dists)))
        misc.log("Max dist: {}".format(np.max(self.dists)))
        misc.log("Mean dist: {}".format(np.mean(self.dists)))

        misc.log("Min azimuth: {}".format(np.min(self.azimuths)))
        misc.log("Max azimuth: {}".format(np.max(self.azimuths)))
        misc.log("Mean azimuth: {}".format(np.mean(self.azimuths)))

        misc.log("Min elev: {}".format(np.min(self.elevs)))
        misc.log("Max elev: {}".format(np.max(self.elevs)))
        misc.log("Mean elev: {}".format(np.mean(self.elevs)))

        misc.log("Min visib fract: {}".format(np.min(self.visib_fracts)))
        misc.log("Max visib fract: {}".format(np.max(self.visib_fracts)))
        misc.log("Mean visib fract: {}".format(np.mean(self.visib_fracts)))

    def plot_stats(self):
        # Visualize distributions.
        plt.figure()
        plt.hist(self.dists, bins=100)
        plt.title("Object distance")

        plt.figure()
        plt.hist(self.azimuths, bins=100)
        plt.title("Azimuth")

        plt.figure()
        plt.hist(self.elevs, bins=100)
        plt.title("Elevation")

        plt.figure()
        plt.hist(self.visib_fracts, bins=100)
        plt.title("Visibility fraction")

        plt.show()

    def convert_to_pandas(self):
        p_dict = dict(
            dists=self.dists,
            azimuths=self.azimuths,
            elevs=self.elevs,
            visib_fracts=self.visib_fracts,
        )
        self.df = pd.DataFrame(p_dict)
        return self.df

    def plot_stats_seaborn(self):
        df = self.convert_to_pandas()
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"Gt distribution (N={self.ims_count})")
        sns.histplot(x="dists", data=df, bins=50, stat="count", kde=True)
        plt.show()


@dataclass
class DataCombiner:
    gt_dist_stats: List[GtDistCalculator]

    def __post_init__(self):
        self.df_list = []
        for gt_stats in self.gt_dist_stats:
            split = gt_stats.get_dp_split()["split"]
            df = gt_stats.convert_to_pandas()
            df["split"] = split
            self.df_list.append(df)

        self.df = pd.concat(self.df_list)

    def get_title(self):
        train_img = self.df.loc[self.df["split"] == "train"].shape[0]
        test_img = self.df.loc[self.df["split"] == "test"].shape[0]

        title = f"Gt distribution (Train={train_img}, Test={test_img})"
        return title

    def plot_stats_seaborn(self):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        fig.suptitle(self.get_title())
        fig.set_tight_layout(True)
        fig.subplots_adjust(**adjust_params)

        stat = "proportion"
        sns.histplot(x="dists", data=self.df, ax=ax[0], bins=50, stat=stat, kde=True, hue="split")
        ax[0].set_xlabel("Distance to camera (mm)")
        sns.histplot(
            x="visib_fracts",
            data=self.df,
            ax=ax[1],
            bins=50,
            stat=stat,
            kde=True,
            hue="split",
        )
        ax[1].set_xlabel("Visibility fraction")

        [ax.grid(axis="y") for ax in ax]
        plt.show()


if __name__ == "__main__":
    # Load train parameters.
    dp_split = dataset_params.get_split_params(
        p["datasets_path"], p["dataset"], "train", p["dataset_split_type"]
    )
    train_gt_stats = GtDistCalculator(dp_split)
    train_gt_stats.calc_gt_dist()
    train_gt_stats.plot_stats_seaborn()

    # Load test parameters
    dp_split = dataset_params.get_split_params(
        p["datasets_path"], p["dataset"], "test", p["dataset_split_type"]
    )
    test_gt_stats = GtDistCalculator(dp_split)
    test_gt_stats.calc_gt_dist()
    test_gt_stats.plot_stats_seaborn()

    stats_combiner = DataCombiner([train_gt_stats, test_gt_stats])
    # stats_combiner.df.to_csv("./gt_distribution.csv")
    stats_combiner.plot_stats_seaborn()
