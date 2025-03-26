"""
@file   argparser.py
@brief  Parsing of command line arguments.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CGP circuit.")
    parser.add_argument("-f", "--file", type=str, required=True, help="The file to load the CGP circuit from.")

    parser.add_argument("-c", "--criterion", type=str, default="area", choices=["area", "error"], help="The criterion for getting fitness (default: area).")
    parser.add_argument("-p", "--population", type=int, default=10, help="The size of the population to use during evolution (default: 10).")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="The number of epochs to run the evolution for (default: 10).")
    parser.add_argument("-m", "--mutation_rate", type=float, default=0.03, help="Mutation rate to be used in evolution (default: 0.03).")
    parser.add_argument("-t", "--tau", type=float, default=10, help="Tau value (max error/area) to be used (default: 10).")

    parser.add_argument("--pretrain", type=int, help="For how long to pretrain the initial population [use this for optimizing error] (default: None).")
    parser.add_argument("--finetune", type=int, help="For how long to finetune the best individual found (default: None).")

    parser.add_argument("-b", "--batch_size", type=int, default=16, help="The batch size used during generation and inference, will be converted to 2 ** batch_size (default: 16).")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], default=None, help="The device to use during inference (if not provided cuda will be tried).")

    parser.add_argument("--log", type=str, default=None, help="Folder to where store logs (default: None).")
    return parser.parse_args()
