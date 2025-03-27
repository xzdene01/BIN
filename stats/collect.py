import os
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations.")

    parser.add_argument("-d", "--dir", type=str, default="logs", help="The directory with the logs to collect stats from (default: logs).")
    parser.add_argument("-c", "--criterion", type=str, default="area", choices=["area", "error"], help="The criterion to collect stats for (default: area).")
    parser.add_argument("-o", "--output", type=str, default=None, help="The output file to save the plot to (default: None).")

    return parser.parse_args()


def main():
    args = parse_args()

    sec_criterion = "error" if args.criterion == "area" else "area"

    stats = {}
    taus = []
    best_areas = []
    print(f"Collecting AREA optimization stats for {sec_criterion}s:", end=" ")
    for folder in os.listdir(args.dir):
        if args.criterion not in folder:
            continue
        
        tau = float(folder.split("_")[1])
        taus.append(tau)
        stats[tau] = []
        print(f"[{tau}]", end=" ")

        best_area = None
        for sub_folder in os.listdir(os.path.join(args.dir, folder)):
            with open(os.path.join(args.dir, folder, sub_folder, "metadata.json"), "r") as f:
                stats[tau].append(json.load(f))
            area = stats[tau][-1]["best_area"]
            error = stats[tau][-1]["best_error"]
            if best_area is None or area < best_area[0]:
                best_area = (area, error)
        best_areas.append(best_area)
    print()

    plt.figure()
    errors = [stat["best_error"] for tau in stats for stat in stats[tau]]
    areas = [stat["best_area"] for tau in stats for stat in stats[tau]]

    # Plot lower bounds for each tau
    best_areas = sorted(best_areas, key=lambda x: x[0])
    best_as = [best_area[0] for best_area in best_areas]
    best_es = [best_area[1] for best_area in best_areas]
    plt.plot(best_es, best_as, "--", color="lightblue", label="Lower area bound")

    # Show constraints
    alpha = 0.4
    if args.criterion == "area":
        for tau in taus:
            plt.axvline(x=tau, color=(1, 0, 0, alpha), linestyle="--")
    else:
        for tau in taus:
            plt.axhline(y=tau, color=(1, 0, 0, alpha), linestyle="--")
    
    # Plot trendline/approximation
    coefs = np.polyfit(errors, areas, 2)
    trendline = np.poly1d(coefs)
    x_fit = np.linspace(min(errors), max(errors), 100)
    y_fit = trendline(x_fit)
    plt.plot(x_fit, y_fit, "--", color="green", label=f"Trendline: {coefs[0]:.2f}x + {coefs[1]:.2f}")


    plt.xlabel("Error")
    plt.ylabel("Area")
    plt.title(f"Error vs. Area (with different tau/{sec_criterion} values)")
    plt.grid()
    plt.legend()

    # Use logarithmic scale on x axis
    plt.xscale("log")

    # Plot all areas and errors in one figure
    plt.plot(errors, areas, "o")

    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()