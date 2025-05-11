# BIN - 1. Aproximace násobiček pomocí CGP

## Informations

- name: Jan Zdeněk
- login: xzdene01
- contact: <xzdene01@vutbr.cz>
- GitHub: <https://github.com/xzdene01/BIN>
- date: 11.5.2025

## Description

This project implements a **Cartesian Genetic Programming (CGP)**–based evolutionary framework for optimizing digital circuits (e.g., multipliers) with respect to two competing objectives:

- **Area**: the size (number of nodes/gates) of the circuit
- **Error**: the functional approximation error compared to the exact circuit

Depending on your choice of primary criterion (`area` or `error`), the algorithm will first evolve a population to meet a threshold ($\tau$) on that criterion, then optionally finetune for the secondary one.

## Features

- **Pretraining** (optional): Quickly drive the circuit to satisfy the secondary criterion before main evolution (this is required when $\tau$ is area - firstly the system need to reduce are below threshold)
- **Evolutionary loop**: Standard CGP with population, mutation and selection
- **Finetuning** (optional): After the main run, optimize the secondary criterion
- **Logging & Visualization**: Tracks best area/error over epochs and can plot/save results

## Requirements

- Conda and Python installed
- (optional for cuda compute) have Cuda installed (12.7) with all requirements

## Installation

To set up the project environment with Conda, simply run:

```bash
# From the project root (where env.yaml lives)
conda env create -f env.yaml

# Activate the new environment
conda activate bin
```

## Running the Evolution

With your Conda environment active and a CGP circuit file prepared, start the evolutionary optimization by running:

```bash
python main.py \
  --file path/to/your_circuit.json \
  --criterion area \
  --population 50 \
  --epochs 100 \
  --mutation_rate 0.05 \
  --tau 5.0 \
  --pretrain 20 \
  --finetune 30 \
  --batch_size 16 \
  --device cuda \
  --log results/ \
  --step 10
```

| Flag              | Short | Default   |Description                                                                        |
| ----------------- | ----- | --------- | --------------------------------------------------------------------------------  |
| `--file`          | `-f`  | required  | Path to file describing the initial CGP circuit (in standard/ArithsGen format).   |
| `--criterion`     | `-c`  | `area`    | Primary optimization target: `area` or `error`.                                   |
| `--population`    | `-p`  | `10`      | Number of individuals per generation.                                             |
| `--epochs`        | `-e`  | `10`      | Total number of evolution epochs.                                                 |
| `--mutation_rate` | `-m`  | `0.03`    | Probability of mutating each gene per epoch.                                      |
| `--tau`           | `-t`  | `10`      | Threshold $\tau$ on the secondary criterion (anything over has $f(c) = \infty$).  |
| `--pretrain`      |       | `0`       | Number of epochs to pretrain on the secondary criterion before the main run.      |
| `--finetune`      |       | `0`       | Number of epochs to finetune the best individual after the main run.              |
| `--batch_size`    | `-b`  | `16`      | Number of test cases in one batch, final will be $2^N$.                           |
| `--device`        | `-d`  | `auto`    | Compute device: `cuda` or `cpu`.                                                  |

### Pretraining

- use only when optimizing for error ($\tau$ is area and needs to be lowered to match max required)
- will only run until the requirement ($\tau$) is matched

## Running visualizations

There are several script for all of the used visualizations. Before running any of these please extract all log files:

```bash
python stats/collect.py -i <LOG_DIR> -o <OUT_DIR>
```

, where `<LOG_DIR>` is specified logging directory (all logs form this will be collected) and `<OUT_DIR>` is arbitrary output directory (might not exist yet) where the final .CSV fle will be saved.

After extracting all log info you might run any of the scripts in stats. When no output os specified all figures will be shown (using `plt.show()`), otherwise all figures will be saved in specifieddirectory.
