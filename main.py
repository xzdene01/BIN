import torch

from circuit.cgp_circuit import CGPCircuit
from utils import argparser
from utils.maybe_tqdm import maybe_tqdm
from genetic.core import Population
from logger.stats_logger import StatsLogger


def main():
    args = argparser.parse_args()

    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CGP circuit from file
    cgp = CGPCircuit(file=args.file)

    # Calculate bits per input
    in_bits = int(cgp.prefix["c_in"] / cgp.prefix["c_ni"])
    print(f"Bits per input: {in_bits}")

    batch_size = 2 ** args.batch_size
    total_trials = int(2 ** (2 * in_bits))
    total_chunks = max(1, total_trials // batch_size)
    print(f"Batch size: {batch_size}", end=", ") # How many inputs to process at once
    print(f"Trials: {total_trials}", end=", ") # How many inputs to process in total
    print(f"Chunks: {total_chunks}") # How many batches to process in total

    print(f"Criterion: {args.criterion}")
    print("Active nodes:", len(cgp))

    # If num epochs > 20 use tqdm on epoch level, else use tqdm on inference level
    inf_tqdm = True if args.epochs <= 20 else False

    # Get initial population from input CGP
    population = Population(args.population,
                            args.mutation_rate,
                            args.criterion,
                            args.tau, cgp,
                            batch_size, do_mut=True,
                            device=device,
                            use_tqdm=inf_tqdm)
    best = None

    logger = StatsLogger(args.criterion, args.tau, args, args.log)

    #################
    # Run evolution #
    #################

    for i in maybe_tqdm(range(args.epochs), use_tqdm=not inf_tqdm, desc="Epochs", unit="epoch"):
        if inf_tqdm:
            print("=========================================")
            print(f"Epoch {i + 1}/{args.epochs}")

        # Recalculate fitnesses of the whole population
        population.calc_fitnesses_vec()

        # Get best individual from population and print stats
        best = population.get_best()
        if inf_tqdm:
            print("Best area:", best.area, ", Best error:", best.error, ", Best fitness:", best.fitness)

        # Log the best area and error
        logger.log(i, best.area, best.error)

        # Init new population for next epoch
        # !!! This will use the best individual and mutate the new population
        population.populate()
    
    if args.finetune:
        ####################################
        # Finetune for secondary criterion #
        ####################################

        # Switch criterion and set current best as boundary
        criterion = "error" if args.criterion == "area" else "area"
        tau = best.area if criterion == "error" else best.error

        # Init new population with secondary criterion
        population = Population(args.population,
                                args.mutation_rate,
                                criterion,
                                tau,
                                best.cgp,
                                batch_size,
                                do_mut=True,
                                device=device,
                                use_tqdm=inf_tqdm)
        best = None

        for i in maybe_tqdm(range(args.finetune), use_tqdm=not inf_tqdm, desc="Finetuning", unit="epoch"):
            if inf_tqdm:
                print("=========================================")
                print(f"Finetune {i + 1}/{args.finetune}")

            # Recalculate fitnesses of the whole population
            population.calc_fitnesses_vec()

            # Get best individual from population and print stats
            best = population.get_best()
            if inf_tqdm:
                print("Best area:", best.area, ", Best error:", best.error, ", Best fitness:", best.fitness)
            
            # Log the best area and error
            logger.log(i, best.area, best.error, finetune=True)

            # Init new population for next epoch
            # !!! This will use the best individual and mutate the new population
            population.populate()
    
    print("=========================================")
    print("Best individual:")
    print("\tArea:", best.area)
    print("\tError:", best.error)

    # Show or save stats saved in logger
    logger.plot_scatter(save=args.log)
    logger.plot(save=args.log)
    if args.log:
        logger.save_logs(cgp=str(best.cgp))
    else:
        print("=========================================")
        print("!!! Best individual was not saved !!!")
        print(str(best.cgp))
        print("=========================================")


if __name__ == "__main__":
    main()
