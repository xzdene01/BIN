import torch
import argparse

from circuit.cgp_circuit import CGPCircuit

def parse_args():
    parser = argparse.ArgumentParser(description="Run the CGP circuit.")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], default=None, help="The device to use during inference.")
    return parser.parse_args()

def main():
    args = parse_args()

    cgp = CGPCircuit(file="generated/arrdiv4.cgp")

    # generate sample 8-bit input
    inputs = torch.randint(0, 2, (8,)).bool()
    cgp.forward(inputs, device=args.device)


if __name__ == "__main__":
    main()