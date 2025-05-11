"""
@file   visualize_queue.py
@brief  Create a visualization AREA vs. ERROR for all runs and show the general trend when changing tau.
@author Jan Zdeněk (xzdene01)
@date   27/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import get_trend_exp, get_trend_poly, set_log_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations. Show parreto queues for area and error with trend lines.")
    parser.add_argument("-s", "--source_file", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    return parser.parse_args()


def plot_queue(df: pd.DataFrame, ax: plt.Axes):
    areas_df = df["best_area"]
    errors_df = df["best_error"]

    # Plot taus
    criterion = df["criterion"].iloc[0]
    taus = df["tau"].unique()
    taus = sorted(taus)
    tau_color = (1, 0, 0, 0.3)
    if criterion == "area":
        for tau in taus:
            ax.axvline(x=tau, color=tau_color, linestyle="--")
    else:
        for tau in taus:
            ax.axhline(y=tau, color=tau_color, linestyle="--")

    # Plot best individuals
    sns.scatterplot(data=df, x="best_error", y="best_area", style="tau", hue="tau", palette="tab10", ax=ax, s=100, legend="full")

    # Connect best individuals from each tau
    extrems_color = (0, 0, 1, 0.5)
    best_from_taus = []
    worst_from_taus = []
    for tau in taus:
        tau_df = df[df["tau"] == tau]
        best_idx = tau_df[f"best_{criterion}"].idxmin()
        best_from_taus.append((errors_df[best_idx], areas_df[best_idx]))
        worst_idx = tau_df[f"best_{criterion}"].idxmax()
        worst_from_taus.append((errors_df[worst_idx], areas_df[worst_idx]))
    ax.plot([best[0] for best in best_from_taus], [best[1] for best in best_from_taus], color=extrems_color, marker="", linestyle="--", label="Extremes from each tau")
    ax.plot([worst[0] for worst in worst_from_taus], [worst[1] for worst in worst_from_taus], color=extrems_color, marker="", linestyle="--")

    # Plot trend line
    trend_color = (0, 0, 0, 0.3)
    if criterion == "area":
        fn, params = get_trend_exp(errors_df, areas_df)
        xs = np.linspace(errors_df.min(), errors_df.max(), num=200)
        ys = fn(xs)
        ax.plot(xs, ys, color=trend_color, linestyle="-", linewidth=20, label=f"Trend line ({params[0]:.3f} * exp(-{params[1]:.3f} * x) + {params[2]:.3f})")
    else:
        fn, params = get_trend_poly(errors_df, areas_df, degree=2)
        coefs = [float(coef) for coef in params[0]]
        coefs_str = ", ".join([f"{coef:.3f}" for coef in coefs])
        xs = np.linspace(errors_df.min(), errors_df.max(), num=200)
        ys = fn(xs)
        ax.plot(xs, ys, color=trend_color, linestyle="-", linewidth=20, label=f"Trend line (coefs: {coefs_str}, intercept: {params[1]:.3f})")
    
    ax.set_xlabel("Errors")
    ax.set_ylabel("Areas")
    ax.legend()
    ax.grid(True)

    ax.set_xscale("log")
    set_log_ticks(ax, 10)

    if criterion == "area":
        ax.set_title("Area optimization method")
    else:
        ax.set_title("Error optimization method")
    ax.grid(False)


def main():
    args = parse_args()

    df = pd.read_csv(args.source_file)
    df_area = df.query("criterion == 'area'")
    df_error = df.query("criterion == 'error'")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    plot_queue(df_area, ax1)
    plot_queue(df_error, ax2)

    fig.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


if __name__ == "__main__":
    main()