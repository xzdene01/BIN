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

    print(f"Criterion: {args.criterion}")

    # Get initial population
    population = Population(args.population, args.mutation_rate, args.criterion, cgp)
    for i in range(args.epochs):
        print("=========================================")
        print(f"Epoch {i + 1}/{args.epochs}")

        # Recalculate fitnesses of the whole population
        population.get_fittness(batch_size, device)

        # Get best individual from population
        best_cgp, best_area, best_error = population.get_best()
        print("Best area:", best_area, ", Best error:", best_error)

        # Mutate the whole population


if __name__ == "__main__":
    main()
