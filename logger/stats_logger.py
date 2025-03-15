import os
from datetime import datetime
import matplotlib.pyplot as plt

class StatsLogger:
    def __init__(self, criterion, tau):
        self.criterion = criterion
        self.tau = tau

        self.iterations = []
        self.areas = []
        self.errors = []

        self.iterations_finetune = []
        self.areas_finetune = []
        self.errors_finetune = []

        self.log_dir = f"logs/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

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

        plt.plot(self.errors, self.areas, color='blue', marker='x', linestyle='--', label="Epochs")

        if len(self.areas_finetune) > 0:
            plt.plot(self.errors_finetune, self.areas_finetune, color='lightblue', marker='x', linestyle='--', label="Finetune")
            plt.plot(
                [self.errors[-1], self.errors_finetune[0]],
                [self.areas[-1], self.areas_finetune[0]],
                color='lightblue', linestyle='--',
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
        plt.plot(self.iterations, self.areas, color='blue', marker='x', linestyle='--', label="Areas")
        if len(self.areas_finetune) > 0:
            plt.plot(self.iterations_finetune, self.areas_finetune, color='lightblue', marker='x', linestyle='--', label="Areas Finetune")
            plt.plot(
                [self.iterations[-1], self.iterations_finetune[0]],
                [self.areas[-1], self.areas_finetune[0]],
                color='lightblue', linestyle='--',
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
        plt.plot(self.iterations, self.errors, color='red', marker='x', linestyle='--', label="Errors")
        if len(self.errors_finetune) > 0:
            plt.plot(self.iterations_finetune, self.errors_finetune, color='lightcoral', marker='x', linestyle='--', label="Errors Finetune")
            plt.plot(
                [self.iterations[-1], self.iterations_finetune[0]],
                [self.errors[-1], self.errors_finetune[0]],
                color='lightcoral', linestyle='--',
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
    
    def save_logs(self):
        os.makedirs(self.log_dir, exist_ok=True)

        with open(os.path.join(self.log_dir, "areas.txt"), "w") as file:
            for i, area in zip(self.iterations, self.areas):
                file.write(f"{i},{area}\n")
            for i, area in zip(self.iterations_finetune, self.areas_finetune):
                file.write(f"{i},{area}\n")
        
        with open(os.path.join(self.log_dir, "errors.txt"), "w") as file:
            for i, error in zip(self.iterations, self.errors):
                file.write(f"{i},{error}\n")
            for i, error in zip(self.iterations_finetune, self.errors_finetune):
                file.write(f"{i},{error}\n")