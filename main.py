"""
@file   main.py
@brief  Main file for running the evolution process. This file is the entry point of the program.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import torch

from circuit.cgp_circuit import CGPCircuit
from utils import argparser
from utils.maybe_tqdm import maybe_tqdm
from genetic.core import Population
from utils.stats_logger import StatsLogger


def main():
    args = argparser.parse_args()
    logger = StatsLogger(args.criterion, args.tau, args, args.log)

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

    # If num epochs <= 5 use tqdm on epoch level, else use tqdm on inference level
    inf_tqdm = True if args.epochs <= 5 else False

    second_criterion = "error" if args.criterion == "area" else "area"
    
    pretrain_count = 0
    if args.pretrain:
        ##################################
        # Pretrain to at least match tau #
        ##################################

        # Init new population with secondary criterion
        population = Population(args.population,
                                args.mutation_rate,
                                second_criterion,
                                float("inf"), # Just get area to reqired tau
                                cgp,
                                batch_size,
                                do_mut=True,
                                device=device,
                                use_tqdm=inf_tqdm)
        best = None
        for i in maybe_tqdm(range(args.pretrain), use_tqdm=not inf_tqdm, desc="Pretraining", unit="epoch"):
            if inf_tqdm:
                print("=========================================")
                print(f"Pretrain {i + 1}/{args.finetune}")

            # Recalculate fitnesses of the whole population
            population.calc_fitnesses_vec()

            # Get best individual from population and print stats
            best = population.get_best()
            if inf_tqdm:
                print("Best area:", best.area, ", Best error:", best.error, ", Best fitness:", best.fitness)
            
            # Log the best area and error
            if i % args.step == 0 or i == args.pretrain - 1:
                logger.log(i, best.area, best.error, flag="pretrain")
            pretrain_count += 1

            if best.fitness <= args.tau:
                cgp = best.cgp
                break

            # Init new population for next epoch
            # !!! This will use the best individual and mutate the new population !!!
            population.populate()
        else:
            print("=========================================")
            print("Pretraining failed!")
            print("=========================================")
            return

    # Get initial population from input CGP
    population = Population(args.population,
                            args.mutation_rate,
                            args.criterion,
                            args.tau, cgp,
                            batch_size, do_mut=True,
                            device=device,
                            use_tqdm=inf_tqdm)
    best = None

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
        if i % args.step == 0 or i == args.epochs - 1:
            logger.log(pretrain_count + i, best.area, best.error, flag="normal")

        # Init new population for next epoch
        # !!! This will use the best individual and mutate the new population !!!
        population.populate()
    
    if args.finetune:
        ####################################
        # Finetune for secondary criterion #
        ####################################

        # Switch criterion and set current best as boundary
        tau = best.area if args.criterion == "area" else best.error

        # Init new population with secondary criterion
        population = Population(args.population,
                                args.mutation_rate,
                                second_criterion,
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
            if i % args.step == 0 or i == args.finetune - 1:
                logger.log(pretrain_count + args.epochs + i, best.area, best.error, flag="finetune")

            # Init new population for next epoch
            # !!! This will use the best individual and mutate the new population !!!
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
