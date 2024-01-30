import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from ERH import ErrorRecordHeader as ERH
import matplotlib.pylab as pylab

params = {
    "figure.titlesize": "xx-large",
    "legend.fontsize": "x-large",
    "figure.figsize": (9, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

adjust_params = dict(top=0.93, bottom=0.088, left=0.125, right=0.9, hspace=0.4, wspace=0.25)


def on_close(event):
    size = event.canvas.figure.get_size_inches()  # Get the size in inches
    print("Resized Figure Size:", size)
    plt.ioff()  # Turn off interactive mode to ensure the script waits for the plot to be closed


error_path = "./scripts/plots_scripts/gdrnet_final_error_metrics.csv"


def main(error_path):
    error_path = Path(error_path)
    assert error_path.exists(), f"Error path {error_path} does not exist"

    df = pd.read_csv(error_path)

    # fig, ax = plt.subplots(1,3)
    # fig.suptitle(f"GDRN error metrics N={df.shape[0]}")
    # fig.set_tight_layout(True)
    # fig.subplots_adjust(**adjust_params)
    # sns.boxplot(df, y="re", ax=ax[0])
    # ax[0].set_ylabel(ERH.re.value)
    # sns.boxplot(df, y="te", ax=ax[1])
    # ax[1].set_ylabel(ERH.te.value)
    # sns.boxplot(df, y="mssd", ax=ax[2])
    # ax[2].set_ylabel(ERH.mssd.name+" (mm)")

    # [a.grid() for a in ax]

    # sns.violinplot(df, y="re", ax=ax[1])

    # sns.swarmplot(df, y="re", ax=ax)
    # sns.swarmplot(df, x="guidance", y="total_errors", color="black", ax=ax, order=["Baseline","Visual","Haptic","Audio"])

    re_median = df["re"].median()
    te_median = df["te"].median()
    mssd_median = df["mssd"].median()

    fig, ax = plt.subplots(3, 1)
    fig.canvas.mpl_connect("close_event", on_close)

    fig.suptitle(f"Test set results (N={df.shape[0]})")
    # fig.set_tight_layout(True)
    fig.set_size_inches(7.24, 7.53)
    fig.subplots_adjust(**adjust_params)
    sns.histplot(df, x="re", ax=ax[0], kde=True, bins=100)
    ax[0].set_xlabel(ERH.re.value)
    ax[0].axvline(re_median, color="red", label="median", linestyle="dashed")
    ax[0].set_xlim(0, 75)
    sns.histplot(df, x="te", ax=ax[1], kde=True, bins=100)
    ax[1].set_xlabel(ERH.te.value)
    ax[1].axvline(te_median, color="red", label="median", linestyle="dashed")
    ax[1].set_xlim(0, 15)
    sns.histplot(df, x="mssd", ax=ax[2], kde=True, bins=100)
    ax[2].set_xlabel(ERH.mssd.name + " (mm)")
    ax[2].axvline(mssd_median, color="red", label="median", linestyle="dashed")
    ax[2].set_xlim(0, 10)

    [a.grid() for a in ax]
    fig.savefig("./scripts/plots_scripts/gdrnet_test_results.png", dpi=fig.dpi)
    plt.show()


if __name__ == "__main__":
    main(error_path)
