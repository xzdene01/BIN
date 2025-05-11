"""
@file   visualize_corel_e.py
@brief  Visualize 2 arbitrary methods even from 2 separate files and visually compare their results.
@author Jan Zdeněk (xzdene01)
@date   29/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import get_trend_exp, get_trend_poly, set_log_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations. Show correlation between two methods.")
    parser.add_argument("-s1", "--source_file1", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-s2", "--source_file2", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-c1", "--criterion1", type=str, required=True, help="The criterion to use for the plot.")
    parser.add_argument("-c2", "--criterion2", type=str, required=True, help="The criterion to use for the plot.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    return parser.parse_args()


def main():
    args = parse_args()

    df1 = pd.read_csv(args.source_file1)
    df1 = df1[["criterion", "best_area", "best_error"]]
    df1 = df1.query("criterion == @args.criterion1")

    df2 = pd.read_csv(args.source_file2)
    df2 = df2[["criterion", "best_area", "best_error"]]
    df2 = df2.query("criterion == @args.criterion2")

    x1 = df1["best_error"]
    y1 = df1["best_area"]

    x2 = df2["best_error"]
    y2 = df2["best_area"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax5 = axs[2, 0]
    ax6 = axs[2, 1]

    x_min = min(x1.min(), x1.min())
    x_max = max(x2.max(), x2.max())

    # set all axs to have the same x and y labels
    for ax in axs.flat:
        ax.set_xlabel("Error")
        ax.set_ylabel("Area")
        ax.set_xscale("log")
        ax.grid()

    # Get curve fit functions
    fit1, _ = get_trend_exp(x1, y1) if args.criterion1 == "area" else get_trend_poly(x1, y1, degree=2)
    fit2, _ = get_trend_exp(x2, y2) if args.criterion2 == "area" else get_trend_poly(x2, y2, degree=2)

    #######################
    # SHOW THE CURVE FITS #
    #######################

    x_pred = np.linspace(x_min, x_max, num=200)
    y_pred1 = fit1(x_pred)
    y_pred2 = fit2(x_pred)

    ax1.scatter(x1, y1, label="Ground-truth", color="blue", marker="x")
    ax1.plot(x_pred, y_pred1, label="Prediction", color="blue")
    ax1.set_title(f"{args.criterion1} vs. {args.criterion1} (to show the curve fits)")
    ax1.legend()


    ax2.scatter(x2, y2, label="Ground-truth", color="red", marker="x")
    ax2.plot(x_pred, y_pred2, label="Prediction", color="red")
    ax2.set_title(f"{args.criterion2} vs. {args.criterion2} (to show the curve fits)")
    ax2.legend()

    #######################################
    # COMPARE ACTUAL AND PREDICTED VALUES #
    #######################################

    ax3.scatter(x1, y1, label="First method", color="blue", marker="x")
    ax3.plot(x_pred, y_pred2, label="Second method prediction", color="red")
    ax3.set_title(f"{args.criterion1} (actual) vs. {args.criterion2} (prediction)")
    ax3.legend()

    ax4.scatter(x2, y2, label="Second method", color="red", marker="x")
    ax4.plot(x_pred, y_pred1, label="First method prediction", color="blue")
    ax4.set_title(f"{args.criterion2} (actual) vs. {args.criterion1} (prediction)")
    ax4.legend()

    ###################################################
    # COMPARE BOTH PREDICTIONS AND BOTH ACTUAL VALUES #
    ###################################################

    ax5.plot(x_pred, y_pred1, label="First method prediction", color="blue")
    ax5.plot(x_pred, y_pred2, label="Second method prediction", color="red")
    ax5.set_title(f"{args.criterion1} (prediction) vs. {args.criterion2} (prediction)")
    ax5.legend()

    ax6.scatter(x1, y1, label="First method", color="blue", marker="x")
    ax6.scatter(x2, y2, label="Second method", color="red", marker="x")
    ax6.set_title(f"{args.criterion1} (actual) vs. {args.criterion2} (actual)")
    ax6.legend()

    for ax in axs.flat:
        set_log_ticks(ax, 10, axis="x", decimals=3)

    fig.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()
    

if __name__ == "__main__":
    main()