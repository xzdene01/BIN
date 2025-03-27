"""
@file   core.py
@brief  Contains the Population class for the genetic algorithm with all necessary methods like:
        - initialization of the population
        - mutation of the population
        - calculation of fitnesses
        - error calculation
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import torch
import copy

from circuit.cgp_circuit import CGPCircuit, opcode_to_str
from utils import generator
from utils.maybe_tqdm import maybe_tqdm
from utils.convert import binary_tensor_to_uint
from .mappings import fitness_functions


class Individual:
    """
    Represents an individual in the population. Is used ONLY for returning the best individual from the population.
    """

    def __init__(self, cgp: CGPCircuit, area: int, error: float, fitness: float):
        self.cgp = cgp
        self.area = area
        self.error = error
        self.fitness = fitness


class Population:
    """
    Represents a population of CGP circuits. The population is initialized with a parent circuit and can be evolved
    using the calc_fitnesses method.
    """

    def __init__(self,
                 size: int,
                 m_rate: float,
                 criterion: str,
                 tau: float,
                 parent: CGPCircuit,
                 batch_size: int,
                 do_mut: bool = False,
                 device="cpu",
                 use_tqdm: bool = False):
        """
        Initialize the population.

        :param size: The size of the population
        :param m_rate: The mutation rate
        :param criterion: The criterion to use for fitness calculation (area or error)
        :param tau: The tau value for fitness calculation (max error/area)
        :param parent: The parent CGP circuit to use as a base
        :param batch_size: The batch size for vectorized operations
        :param do_mut: Whether to mutate the initial population
        :param device: The device to use
        :param use_tqdm: Whether to use tqdm for progress bars
        """
        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = use_tqdm

        self.size = size
        self.m_rate = m_rate
        self.criterion = criterion
        self.tau = tau

        self.c_in = parent.prefix["c_in"]
        self.c_ni = parent.prefix["c_ni"]

        self.best = copy.deepcopy(parent)
        self.best_area = None
        self.best_error = None
        self.best_fitness = None

        # Create initial population and with mutation?
        self.population = []
        self.populate(self.best, do_mut=do_mut)

        # Fitnesses are yet to be calculated
        self.areas = torch.empty(size, device=self.device)
        self.errors = torch.empty(size, device=self.device)
        self.fitnesses = torch.empty(size, device=self.device)

        # Pre-compute weights for conversion between binary and integer representations of outputs
        self.weights_out = 2 ** torch.arange(parent.prefix["c_out"] - 1, -1, -1, device=self.device, dtype=torch.long)

    def populate(self, parent: CGPCircuit = None, do_mut: bool = True):
        """
        Populate the population with new individuals.

        :param parent: The parent to use for the new population (if None, the best individual is used)
        :param do_mut: Whether to mutate the new population
        """
        # Create new population either from parent or best
        if not parent:
            parent = self.best
        self.population = [copy.deepcopy(parent) for _ in range(self.size)]

        # Mutate the new population
        if do_mut:
            self.mutate()

        # Reset all tensors, they are not valid in this new population
        self.areas = torch.empty(self.size, device=self.device)
        self.errors = torch.empty(self.size, device=self.device)
        self.fitnesses = torch.empty(self.size, device=self.device)

    def mutate(self, m_rate: float = None):
        """
        Mutate the population.

        :param m_rate: The mutation rate to use (if None, the default mutation rate is used)
        """
        # !!! Does not save the mutation rate into attribute, that must be done manualy
        if not m_rate:
            m_rate = self.m_rate

        # Mutate everyone except the first one (best)
        for individual in self.population[1:]:

            # Choose random nodes from active one and mutate
            idxs = torch.arange(len(individual.core), device=self.device)
            num_mutations = min(int(m_rate * len(idxs)), len(idxs))
            weights = torch.ones(len(idxs), device=self.device)
            mutation_idxs = torch.multinomial(weights, num_mutations, replacement=False)
            self.mutate_core(individual, idxs[mutation_idxs])

            # Choose random outputs and mutate
            num_mutations = min(int(m_rate * len(individual.outputs)), len(individual.outputs))

            if num_mutations > 0:
                weights = torch.ones(len(individual.outputs), device=self.device)
                mutation_idxs = torch.multinomial(weights, num_mutations, replacement=False)

            # If num of mutations is 0, we need to go stochastic
            else:
                mutation_idxs = []
                for i in range(len(individual.outputs)):
                    if torch.rand(1).item() < m_rate:
                        mutation_idxs.append(i)
                mutation_idxs = torch.tensor(mutation_idxs, device=self.device)

            self.mutate_outputs(individual, mutation_idxs)

    def mutate_core(self, individual: CGPCircuit, mutation_indices: torch.Tensor):
        """
        Mutate the core of the individual.

        :param individual: The individual to mutate
        :param mutation_indices: The indices of the nodes
        """
        for idx in mutation_indices:
            # Choose what to mutate (opcode, in_1 or in_2)
            rnd_num = torch.rand(1).item()
            if rnd_num < 0.3:
                individual.core[idx].op_code = torch.randint(len(opcode_to_str), (1,)).item()
            elif rnd_num < 0.65:
                individual.core[idx].in_1 = torch.randint(self.c_in + len(individual.core), (1,)).item()
            else:
                individual.core[idx].in_2 = torch.randint(self.c_in + len(individual.core), (1,)).item()

    def mutate_outputs(self, individual: CGPCircuit, mutation_indices: torch.Tensor):
        """
        Mutate the outputs of the individual.

        :param individual: The individual to mutate
        :param mutation_indices: The indices of the outputs
        """
        for idx in mutation_indices:
            individual.outputs[idx] = torch.randint(self.c_in + len(individual.core), (1,)).item()

    def calc_fitnesses(self) -> torch.Tensor:
        """
        Calculate the fitnesses of the population.

        :return: The fitnesses of the population
        """
        for i, individual in enumerate(maybe_tqdm(self.population, use_tqdm=self.use_tqdm, desc="Calculating fitness", unit="individual")):
            self.areas[i] = len(individual)
            self.errors[i] = self.get_error(individual)
        self.fitnesses = fitness_functions[self.criterion](self.areas, self.errors, self.tau)

        # !!! Current best must always be at index 0
        # best_idx, _ = min(enumerate(self.fitnesses), key=lambda pair: (pair[1], len(self.fitnesses) - pair[0]))
        best_idx = None
        best_fitness = None
        for i, fitness in enumerate(self.fitnesses):
            if best_fitness is None or fitness <= best_fitness:
                best_idx = i
                best_fitness = fitness

        self.best = copy.deepcopy(self.population[best_idx])
        self.best_area = int(self.areas[best_idx].item())
        self.best_error = self.errors[best_idx].item()
        self.best_fitness = self.fitnesses[best_idx].item()

        return self.fitnesses

    def get_error(self, individual: CGPCircuit) -> float:
        """
        Get the error of the individual.

        :param individual: The individual to get the error for
        :return: The error of the individual
        """
        in_bits = int(self.c_in / self.c_ni)
        total_error = 0.0

        for in1, in2, ref in generator.generate_all_vec(in_bits, self.batch_size, device=self.device):
            batch_input = torch.cat((in1, in2), dim=1).to(self.device)
            batch_output = individual.forward_batch(batch_input, device=self.device)

            output_int = (batch_output.long() * self.weights_out).sum(dim=1)
            ref_int = (ref.long() * self.weights_out).sum(dim=1)

            total_error += torch.abs(output_int - ref_int).sum().item()

        total_samples = 2 ** self.c_in
        return total_error / total_samples

    def calc_fitnesses_vec(self) -> torch.Tensor:
        """
        Calculate the fitnesses of the population using vectorized operations.

        :return: The fitnesses of the population
        """
        self.areas = torch.tensor([len(ind) for ind in self.population], device=self.device).float()
        self.errors = self.get_error_vec()
        self.fitnesses = fitness_functions[self.criterion](self.areas, self.errors, self.tau)

        # !!! Current best must always be at index 0
        b_val_all, _ = torch.min(self.fitnesses, dim=0)
        b_val_tail, b_idx_tail = torch.min(self.fitnesses[1:], dim=0)
        best_idx = 0 if b_val_all < b_val_tail else b_idx_tail + 1

        self.best = copy.deepcopy(self.population[best_idx])
        self.best_area = int(self.areas[best_idx].item())
        self.best_error = self.errors[best_idx].item()
        self.best_fitness = self.fitnesses[best_idx].item()

        return self.fitnesses

    def get_error_vec(self, individuals: list[CGPCircuit] = None) -> torch.Tensor:
        """
        Get the error of many individuals (or whole population) at once.

        :param individuals: List of individuals to get the error for (if None, the whole population is used)
        :return: Errors of selected individuals
        """
        if not individuals:
            individuals = self.population

        num_ind = len(individuals)
        in_bits = int(self.c_in / self.c_ni)
        total_errors = torch.zeros(num_ind, device=self.device)

        for in1, in2, ref in generator.generate_all_vec(in_bits, self.batch_size, device=self.device):
            # shape [in1, in2]: (batch_size, c_in / c_ni)
            # shape [ref]: (batch_size, c_out)

            # shape: (batch_size, c_in)
            batch_inputs = torch.cat((in1, in2), dim=1)

            # shape: (N, batch_size, c_out)
            batch_outputs = torch.stack([ind.forward_batch(batch_inputs, device=self.device) for ind in individuals])

            # shape: (N, batch_size)
            outputs_int = (batch_outputs.long() * self.weights_out).sum(dim=2)

            # shape: (batch_size)
            ref_int = (ref.long() * self.weights_out).sum(dim=1)

            # shape: (N, batch_size)
            batch_errors = torch.abs(outputs_int - ref_int.unsqueeze(0))

            # shape: (N)
            total_errors += batch_errors.sum(dim=1)

        total_samples = 2 ** self.c_in
        return total_errors / total_samples

    def get_best(self) -> Individual:
        """
        Get the best individual from the population."

        :return: The best individual
        """
        return Individual(self.best, self.best_area, self.best_error, self.best_fitness)
