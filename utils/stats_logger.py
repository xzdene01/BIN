"""
@file   stats_logger.py
@brief  Logger for logging statistics of the training process.
        This is more of a helper class with 0 effect on the actual training process.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from stats.helpers import set_log_ticks

class StatsLogger:
    def __init__(self, criterion, tau, args, folder=None):
        self.criterion = criterion
        self.tau = tau

        self.metadata = {
            "file": args.file,
            "criterion": criterion,
            "population": args.population,
            "epochs": args.epochs,
            "mutation_rate": args.mutation_rate,
            "tau": tau,
            "pretrain": args.pretrain,
            "finetune": args.finetune,
            "batch_size": args.batch_size,
            "device": args.device,
            "log": args.log,
        }

        self.iterations = []
        self.areas = []
        self.errors = []
        self.flags = []

        if not folder:
            folder = "logs"
        self.log_dir = f"{folder}/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

    def log(self, iteration, area, error, flag):
        self.iterations.append(iteration)
        self.areas.append(area)
        self.errors.append(error)

        if flag not in ["pretrain", "normal", "finetune"]:
            raise ValueError(f"Invalid flag: {flag}, must be one of ['pretrain', 'normal', 'finetune']")
        self.flags.append(flag)
    
    def plot_scatter(self, save=False):
        len_pretrain = len([flag for flag in self.flags if flag == "pretrain"])
        len_finetune = len([flag for flag in self.flags if flag == "finetune"])

        plt.figure()

        # Plot pretrain
        if len_pretrain > 0:
            plt.plot(
                self.errors[:len_pretrain],
                self.areas[:len_pretrain],
                color="orange", marker="", linestyle="--", label="Pretrain", linewidth=2
            )
            plt.plot(
                [self.errors[len_pretrain-1], self.errors[len_pretrain]],
                [self.areas[len_pretrain-1], self.areas[len_pretrain]],
                color="orange", marker="", linestyle="--", linewidth=2
            )
        
        # Plot normal
        if len_finetune == 0:
            plt.plot(
                self.errors[len_pretrain:],
                self.areas[len_pretrain:],
                color="blue", marker="", linestyle="-", label="Train", linewidth=2
            )
        else:
            plt.plot(
                self.errors[len_pretrain:-len_finetune],
                self.areas[len_pretrain:-len_finetune],
                color="blue", marker="", linestyle="-", label="Train", linewidth=2
            )

        # Plot finetune
        if len_finetune > 0:
            plt.plot(
                self.errors[-len_finetune:],
                self.areas[-len_finetune:],
                color="gray", marker="", linestyle="--", label="Finetune", linewidth=2
            )
            plt.plot(
                [self.errors[-len_finetune-1], self.errors[-len_finetune]],
                [self.areas[-len_finetune-1], self.areas[-len_finetune]],
                color="gray", marker="", linestyle="--", linewidth=2
            )
        
        # Show end point
        plt.plot(
            self.errors[-1],
            self.areas[-1],
            color="green", marker="o", linestyle=""
        )

        # Plot bounds
        if self.criterion == "area":
            plt.axvline(x=self.tau, color='r', linestyle='--', label=f"Error bound")
        else:
            plt.axhline(y=self.tau, color='r', linestyle='--', label=f"Area bound")
        
        plt.xlabel("Error")
        plt.ylabel("Area")
        plt.title("Area vs Error")
        plt.legend()
        plt.grid(True)

        plt.xscale("log")
        ax = plt.gca()
        set_log_ticks(ax, 10, axis="x")

        if save:
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, "scatter.png"))
        else:
            try:
                plt.show()
            except Exception:
                print("Cannot show plot. Save it instead with --log flag.")

    def plot(self, save=False):
        len_pretrain = len([flag for flag in self.flags if flag == "pretrain"])
        len_finetune = len([flag for flag in self.flags if flag == "finetune"])

        plt.figure()

        #########
        # Areas #
        #########

        plt.subplot(3, 1, 1)

        # Pretrain
        if len_pretrain > 0:
            plt.plot(self.iterations[:len_pretrain], self.areas[:len_pretrain], color='orange', marker='', linestyle='--', label="Pretrain", linewidth=2)
        
        # Train
        if len_finetune == 0:
            plt.plot(self.iterations[len_pretrain:], self.areas[len_pretrain:], color='blue', marker='', linestyle='-', label="Train", linewidth=2)
        else:
            plt.plot(self.iterations[len_pretrain:-len_finetune], self.areas[len_pretrain:-len_finetune], color='blue', marker='', linestyle='-', label="Train", linewidth=2)
        
        # Finetune
        if len_finetune > 0:
            plt.plot(self.iterations[-len_finetune:], self.areas[-len_finetune:], color='gray', marker='', linestyle='--', label="Finetune", linewidth=2)
        
        # Show end point
        plt.plot(self.iterations[-1], self.areas[-1], color='green', marker='o', linestyle='')

        # Show bounds
        if self.criterion == "error":
            plt.axhline(y=self.tau, color='r', linestyle='--', label=f"Area bound")
        
        plt.xlabel("Iterations")
        plt.ylabel("Area")
        plt.title("Areas")
        plt.legend()
        plt.grid(True)
        
        ##########
        # Errors #
        ##########

        plt.subplot(3, 1, 3)

        # Pretrain
        if len_pretrain > 0:
            plt.plot(self.iterations[:len_pretrain], self.errors[:len_pretrain], color='orange', marker='', linestyle='--', label="Pretrain", linewidth=2)
        
        # Train
        if len_finetune == 0:
            plt.plot(self.iterations[len_pretrain:], self.errors[len_pretrain:], color='blue', marker='', linestyle='-', label="Train", linewidth=2)
        else:
            plt.plot(self.iterations[len_pretrain:-len_finetune], self.errors[len_pretrain:-len_finetune], color='blue', marker='', linestyle='-', label="Train", linewidth=2)
        
        # Finetune
        if len_finetune > 0:
            plt.plot(self.iterations[-len_finetune:], self.errors[-len_finetune:], color='gray', marker='', linestyle='--', label="Finetune", linewidth=2)
        
        # Show end point
        plt.plot(self.iterations[-1], self.errors[-1], color='green', marker='o', linestyle='')
        
        # Show bounds
        if self.criterion == "area":
            plt.axhline(y=self.tau, color='r', linestyle='--', label=f"Error bound")

        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Errors")
        plt.legend()
        plt.grid(True)

        plt.yscale("log")
        ax = plt.gca()
        set_log_ticks(ax, 3, axis="y")

        if save:
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, "areas_errors.png"))
        else:
            try:
                plt.show()
            except Exception:
                print("Cannot show plot. Save it instead with --log flag.")
    
    def save_logs(self, cgp: str = None):
        os.makedirs(self.log_dir, exist_ok=True)

        # save iterations, areas, errors as csv
        df = pd.DataFrame({
            "iteration": self.iterations,
            "area": self.areas,
            "error": self.errors,
            "flag": self.flags
        })
        df.to_csv(os.path.join(self.log_dir, "log.csv"), index=False)
        
        self.metadata["best_area"] = self.areas[-1]
        self.metadata["best_error"] = self.errors[-1]
        with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
            f.write(json.dumps(self.metadata, indent=4))
        
        if cgp:
            with open(os.path.join(self.log_dir, "best.cgp"), "w") as f:
                f.write(cgp)