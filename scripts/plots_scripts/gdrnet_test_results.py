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


    re_median = df["re"].median()
    te_median = df["te"].median()
    mssd_median = df["mssd"].median()




if __name__ == "__main__":
    main(error_path)
