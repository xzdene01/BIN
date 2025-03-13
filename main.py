import torch

from circuit.cgp_circuit import CGPCircuit
from utils import argparser
from genetic.core import Population

def main():
    args = argparser.parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cgp = CGPCircuit(file=args.file)

    in_bits = int(cgp.prefix["c_in"] / cgp.prefix["c_ni"])
    print(f"Bits per input: {in_bits}")

    batch_size = 2 ** args.batch_size
    total_trials = int(2 ** (2 * in_bits))
    total_chunks = max(1, total_trials // batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Total number of trials: {total_trials}")
    print(f"Total chunks: {total_chunks}")

    for i in range(args.epochs):
        print("=========================================")
        print(f"Epoch {i + 1}/{args.epochs}")

        population = Population(args.population, 0.1, cgp)
        fitnesses = population.get_fittness(batch_size, device)

        print(f"Best fitness: {max(fitnesses)}")

if __name__ == "__main__":
    main()