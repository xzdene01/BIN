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
    print(f"Batch size: {batch_size}", end=", ")
    print(f"Trials: {total_trials}", end=", ")
    print(f"Chunks: {total_chunks}")

    print(f"Criterion: {args.criterion}")

    mask = cgp.get_active_mask()
    print("Active nodes:", int(mask.sum().item()))

    # Get initial population
    population = Population(args.population, args.mutation_rate, args.criterion, cgp, mutate=True)

    # Run evolution
    for i in range(args.epochs):
        print("=========================================")
        print(f"Epoch {i + 1}/{args.epochs}")

        # Recalculate fitnesses of the whole population
        population.get_fittness(batch_size, device)

        # Get best individual from population
        best_individual, best_area, best_error = population.get_best()
        print("Best area:", best_area, ", Best error:", best_error)

        # Init new population
        population.populate(best_individual, mutate=True)


if __name__ == "__main__":
    main()
