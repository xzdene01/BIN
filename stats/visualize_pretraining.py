"""
@file   visualize_pretraining.py
@brief  Show the relation between pretraining iterations, pretraining error and final/best error.
@author Jan Zdeněk (xzdene01)
@date   29/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import set_log_ticks

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit pretraining (pretraining iterations, pretraining error and final/best error).")
    parser.add_argument("-s", "--source_file", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.source_file)
    df = df.query("criterion == 'error'")
    
    x1 = []
    x2 = []
    y = []
    labels = []
    for _, row in df.iterrows():
        run_df = pd.read_csv(row["run_log"])

        pretrain_iters = run_df.query("flag == 'normal'")["iteration"].min()
        pretrain_error = run_df.query("flag == 'normal'")["error"].max()
        best_error = row["best_error"]
        label = row["tau"]

        x1.append(pretrain_iters)
        x2.append(pretrain_error)
        y.append(best_error)
        labels.append(label)
    
    data_df = pd.DataFrame({"pretrain_iters": x1, "pretrain_error": x2, "final_error": y, "tau": labels})
    
    fig, (ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))

    sns.scatterplot(data=data_df, x="pretrain_iters", y="final_error", hue="tau", palette="tab10", s=100, style="tau", legend="full", ax=ax1)
    ax1.set_xlabel("Pretraining iterations")
    ax1.set_ylabel("Best error")

    sns.scatterplot(data=data_df, x="pretrain_error", y="final_error", hue="tau", palette="tab10", s=100, style="tau", legend="full", ax=ax2)
    ax2.set_xlabel("Pretraining error")
    ax2.set_ylabel("Best error")

    # sns.scatterplot(data=data_df, x="pretrain_error", y="pretrain_iters", hue="tau", palette="tab10", s=100, style="tau", legend="full", ax=ax3)
    # ax3.set_xlabel("Pretraining error")
    # ax3.set_ylabel("Pretraining iterations")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        set_log_ticks(ax, 10, "x", 0)

        ax.set_yscale("log")
        set_log_ticks(ax, 10, "y", 0)

        ax.legend()

    fig.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


if __name__ == "__main__":
    main()