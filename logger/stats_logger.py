import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

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

        self.iterations_finetune = []
        self.areas_finetune = []
        self.errors_finetune = []

        if not folder:
            folder = "logs"
        self.log_dir = f"{folder}/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

    def log(self, iteration, area, error, finetune=False):
        if finetune:
            self.iterations_finetune.append(iteration + len(self.iterations))
            self.areas_finetune.append(area)
            self.errors_finetune.append(error)
        else:
            self.iterations.append(iteration)
            self.areas.append(area)
            self.errors.append(error)
    
    def plot_scatter(self, save=False):
        plt.figure()

        plt.plot(self.errors, self.areas, color='blue', marker='', linestyle='-', label="Train")

        if len(self.areas_finetune) > 0:
            plt.plot(self.errors_finetune, self.areas_finetune, color='lightblue', marker='', linestyle='-', label="Finetune")
            plt.plot(
                [self.errors[-1], self.errors_finetune[0]],
                [self.areas[-1], self.areas_finetune[0]],
                color='lightblue', linestyle='-',
            )
            plt.plot(self.errors_finetune[-1], self.areas_finetune[-1], color='green', marker='o', label="Final")
        else:
            plt.plot(self.errors[-1], self.areas[-1], color='green', marker='o', label="Final")

        if self.criterion == "area":
            plt.axvline(x=self.tau, color='r', linestyle='--', label=f"Error bound")
        
        plt.xlabel("Error")
        plt.ylabel("Area")
        plt.title("Area vs Error")
        plt.legend()
        plt.grid(True)

        if save:
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, "scatter.png"))
        else:
            plt.show()

    def plot(self, save=False):
        plt.figure()

        # Areas
        plt.subplot(3, 1, 1)
        plt.plot(self.iterations, self.areas, color='blue', marker='', linestyle='-', label="Areas")
        if len(self.areas_finetune) > 0:
            plt.plot(self.iterations_finetune, self.areas_finetune, color='lightblue', marker='', linestyle='-', label="Areas Finetune")
            plt.plot(
                [self.iterations[-1], self.iterations_finetune[0]],
                [self.areas[-1], self.areas_finetune[0]],
                color='lightblue', linestyle='-',
            )
            plt.plot(self.iterations_finetune[-1], self.areas_finetune[-1], color='green', marker='o', label="Final")
        else:
            plt.plot(self.iterations[-1], self.areas[-1], color='green', marker='o', label="Final")
        plt.xlabel("Iterations")
        plt.ylabel("Area")
        plt.title("Areas")
        plt.legend()
        plt.grid(True)

        # Errors
        plt.subplot(3, 1, 3)
        plt.plot(self.iterations, self.errors, color='red', marker='', linestyle='-', label="Errors")
        if len(self.errors_finetune) > 0:
            plt.plot(self.iterations_finetune, self.errors_finetune, color='lightcoral', marker='', linestyle='-', label="Errors Finetune")
            plt.plot(
                [self.iterations[-1], self.iterations_finetune[0]],
                [self.errors[-1], self.errors_finetune[0]],
                color='lightcoral', linestyle='-',
            )
            plt.plot(self.iterations_finetune[-1], self.errors_finetune[-1], color='green', marker='o', label="Final")
        else:
            plt.plot(self.iterations[-1], self.errors[-1], color='green', marker='o', label="Final")
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Errors")
        plt.legend()
        plt.grid(True)

        if save:
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, "areas_errors.png"))
        else:
            plt.show()
    
    def save_logs(self, cgp: str = None):
        os.makedirs(self.log_dir, exist_ok=True)

        with open(os.path.join(self.log_dir, "areas.txt"), "w") as f:
            for i, area in zip(self.iterations, self.areas):
                f.write(f"{i},{area}\n")
            for i, area in zip(self.iterations_finetune, self.areas_finetune):
                f.write(f"{i},{area}\n")
        
        with open(os.path.join(self.log_dir, "errors.txt"), "w") as f:
            for i, error in zip(self.iterations, self.errors):
                f.write(f"{i},{error}\n")
            for i, error in zip(self.iterations_finetune, self.errors_finetune):
                f.write(f"{i},{error}\n")
        
        self.metadata["best_area"] = self.areas[-1] if len(self.areas_finetune) == 0 else self.areas_finetune[-1]
        self.metadata["best_error"] = self.errors[-1] if len(self.errors_finetune) == 0 else self.errors_finetune[-1]
        with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
            f.write(json.dumps(self.metadata, indent=4))
        
        if cgp:
            with open(os.path.join(self.log_dir, "best.cgp"), "w") as f:
                f.write(cgp)