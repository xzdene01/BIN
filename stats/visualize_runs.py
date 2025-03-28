import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations.")
    parser.add_argument("-s", "--source_file", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    parser.add_argument("-c", "--criterion", type=str, required=True, help="The criterion to use for the plot.")
    parser.add_argument("-t", "--tau_index", type=int, default=0, help="The index of the tau to visualize.")
    parser.add_argument("-i", "--confidence_interval", type=float, default=50, help="The confidence interval to visualize.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.source_file)
    df = df.query("criterion == @args.criterion")

    # Filter just relevant tau
    unique_taus = df["tau"].unique()
    taus_sorted = sorted(unique_taus)
    if args.tau_index >= len(taus_sorted):
        raise ValueError(f"Invalid tau index: {args.tau_index}, max index: {len(taus_sorted) - 1}")
    tau = taus_sorted[args.tau_index]
    df = df[df["tau"] == tau]

    # Get all runs
    runs_df = pd.DataFrame()
    runs = df["run_log"].unique()
    for idx, run in enumerate(runs):
        run_df = pd.read_csv(run)
        run_df["run_id"] = idx
        runs_df = pd.concat([runs_df, run_df])
    
    runs_df = runs_df.sort_values(by=["run_id", "iteration"])

    min_iter = max(1, runs_df["iteration"].min())
    max_iter = runs_df["iteration"].max()
    iters_selected = np.linspace(min_iter, max_iter, num=100)

    interpolated = []
    for run_id in runs_df["run_id"].unique():
        run = runs_df[runs_df["run_id"] == run_id].copy()

        x = run["iteration"].values
        y1 = run["area"].values
        y2 = run["error"].values

        f1 = interp1d(x, y1, kind="previous", fill_value="extrapolate")
        f2 = interp1d(x, y2, kind="previous", fill_value="extrapolate")

        y1_interpolated = f1(iters_selected)
        y2_interpolated = f2(iters_selected)

        df_interpolated = pd.DataFrame({
            "iteration": iters_selected,
            "area": y1_interpolated,
            "error": y2_interpolated,
            "run_id": run_id
        })
        interpolated.append(df_interpolated)

    interpolated_df = pd.concat(interpolated)

    _, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1 = sns.lineplot(data=interpolated_df, x="iteration", y="area", ax=ax1,  label="Area",
                       errorbar=("ci", args.confidence_interval), legend=False, color="blue")
    ax2 = sns.lineplot(data=interpolated_df, x="iteration", y="error", ax=ax2,label="Error",
                       errorbar=("ci", args.confidence_interval), legend=False, color="red")

    # Combine legends
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    ax2.legend(handles_ax1 + handles_ax2, labels_ax1 + labels_ax2)
    
    ax1.set_ylabel("Area")
    ax1.set_xlabel("Iteration")
    ax2.set_ylabel("Error")

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    plt.title(f"Tau: {tau}, confidence interval: {args.confidence_interval}%")

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()

if __name__ == "__main__":
    main()