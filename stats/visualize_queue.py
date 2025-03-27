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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations.")
    parser.add_argument("-s", "--source_file", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    parser.add_argument("-c", "--criterion", type=str, required=True, help="The criterion to use for the plot.")
    return parser.parse_args()


def neg_exp_offset(x, a, b, c):
    return c + a * np.exp(-b * x)


def get_trend_exp(x_series, y_series, num_points=200):
    x_data = x_series.to_numpy()
    y_data = y_series.to_numpy()

    # Initial guess for the parameters
    a_guess = y_data[0] - y_data[-1]
    b_guess = 1.0
    c_guess = y_data[-1]
    initial_guess = [a_guess, b_guess, c_guess]

    # Fit the function to the data
    popt, _ = curve_fit(neg_exp_offset, x_data, y_data, p0=initial_guess)
    a_fit, b_fit, c_fit = popt

    # Create a smooth range of x-values for the fitted curve
    xs = np.linspace(x_data.min(), x_data.max(), num_points)
    ys = neg_exp_offset(xs, a_fit, b_fit, c_fit)
    
    return xs, ys, (a_fit, b_fit, c_fit)


def main():
    args = parse_args()

    df = pd.read_csv(args.source_file)
    df = df.query("criterion == @args.criterion")

    areas_df = df["best_area"]
    errors_df = df["best_error"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

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
    ax.plot(errors_df, areas_df, color="black", marker="x", linestyle="", label="Areas")

    # Connect best individuals from each tau
    extrems_color = "blue"# (0, 1, 1, 0.5)
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

    # Plot trend line with seaborn
    trend_color = (0, 0, 0, 0.3)
    xs, ys, params = get_trend_exp(errors_df, areas_df)
    ax.plot(xs, ys, color=trend_color, linestyle="-", linewidth=20, label=f"Trend line ({params[0]:.3f} * exp(-{params[1]:.3f} * x) + {params[2]:.3f})")
    

    # Set errors to log scale
    ax.set_xscale("log")

    ax.set_xlabel("Errors")
    ax.set_ylabel("Areas")
    ax.set_title("Areas vs Errors")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()


if __name__ == "__main__":
    main()