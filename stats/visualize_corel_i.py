"""
@file   visualize_corel_i.py
@brief  Visualize area VS error method from 1 log file and visually compare their results.
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
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations. Show correlation between two methods from one logs file.")
    parser.add_argument("-s", "--source_file", type=str, required=True, help="The file to load the stats from.")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="The file to save the plot to.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.source_file)
    df = df[["criterion", "best_area", "best_error"]]

    df_area = df.query("criterion == 'area'")
    df_error = df.query("criterion == 'error'")

    x_area = df_area["best_error"]
    y_area = df_area["best_area"]

    x_error = df_error["best_error"]
    y_error = df_error["best_area"]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # 2x2 grid of axs
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax5 = axs[2, 0]
    ax6 = axs[2, 1]

    x_min = min(x_area.min(), x_error.min())
    x_max = max(x_area.max(), x_error.max())

    # set all axs to have the same x and y labels
    for ax in axs.flat:
        ax.set_xlabel("Error")
        ax.set_ylabel("Area")
        ax.set_xscale("log")
        ax.grid()

    # Get curve fit functions
    fit_area, _ = get_trend_exp(x_area, y_area)
    fit_error, _ = get_trend_poly(x_error, y_error, degree=2)

    #######################
    # SHOW THE CURVE FITS #
    #######################

    x_pred = np.linspace(x_min, x_max, num=200)
    y_pred_area = fit_area(x_pred)
    y_pred_error = fit_error(x_pred)

    ax1.scatter(x_area, y_area, label="Ground-truth", color="blue", marker="x")
    ax1.plot(x_pred, y_pred_area, label="Prediction", color="blue")
    ax1.set_title("AREA vs. AREA (to show the curve fits)")
    ax1.axvline(x=20, color="black", linestyle="--", label="Max tau error: 20%")
    ax1.axhline(y=15, color="black", linestyle="--", label="Area: 15")
    ax1.legend()


    ax2.scatter(x_error, y_error, label="Ground-truth", color="red", marker="x")
    ax2.plot(x_pred, y_pred_error, label="Prediction", color="red")
    ax2.set_title("ERROR vs. ERROR (to show the curve fits)")
    ax2.axvline(x=50, color="black", linestyle="--", label="Error: 50%")
    ax2.axhline(y=20, color="black", linestyle="--", label="Min tau area: 20")
    ax2.legend()

    #######################################
    # COMPARE ACTUAL AND PREDICTED VALUES #
    #######################################

    ax3.scatter(x_area, y_area, label="Area method", color="blue", marker="x")
    ax3.plot(x_pred, y_pred_error, label="Error method prediction", color="red")
    ax3.set_title("AREA (actual) vs. ERROR (prediction)")
    ax3.legend()

    ax4.scatter(x_error, y_error, label="Error method", color="red", marker="x")
    ax4.plot(x_pred, y_pred_area, label="Area method prediction", color="blue")
    ax4.set_title("ERROR (actual) vs. AREA (prediction)")
    ax4.legend()

    ###################################################
    # COMPARE BOTH PREDICTIONS AND BOTH ACTUAL VALUES #
    ###################################################

    ax5.plot(x_pred, y_pred_area, label="Area method prediction", color="blue")
    ax5.plot(x_pred, y_pred_error, label="Error method prediction", color="red")
    ax5.set_title("AREA (prediction) vs. ERROR (prediction)")
    ax5.axvline(x=50, color="black", linestyle="--", label="Error: 50%")
    ax5.axhline(y=15, color="black", linestyle="--", label="Area: 15")
    ax5.legend()

    ax6.scatter(x_area, y_area, label="Area method", color="blue", marker="x")
    ax6.scatter(x_error, y_error, label="Error method", color="red", marker="x")
    ax6.set_title("AREA (actual) vs. ERROR (actual)")
    ax6.axvline(x=20, color="black", linestyle="--", label="Max tau error: 20%")
    ax6.axhline(y=20, color="black", linestyle="--", label="Min tau area: 20")
    ax6.legend()

    for ax in axs.flat:
        set_log_ticks(ax, 10)

    fig.tight_layout()

    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()
    

if __name__ == "__main__":
    main()