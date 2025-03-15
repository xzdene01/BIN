import torch
import tqdm
import copy

from circuit.cgp_circuit import CGPCircuit, opcode_to_str
from utils import generator
from utils.maybe_tqdm import maybe_tqdm

class Individual:
    def __init__(self, cgp: CGPCircuit, area: int, error: float, fitness: float):
        self.cgp = cgp
        self.area = area
        self.error = error
        self.fitness = fitness


class Population:
    def __init__(self, size: int, m_rate: float, criterion: str, tau: float, parent: CGPCircuit, batch_size: int, do_mut: bool = False, device="cpu", use_tqdm: bool = False):
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

        self.population = []
        self.populate(self.best, do_mut=do_mut)

        # Fitnesses are yet to be calculated
        self.areas = torch.empty(size, device=self.device)
        self.errors = torch.empty(size, device=self.device)
        self.fitnesses = torch.empty(size, device=self.device)

    def populate(self, parent: CGPCircuit = None, do_mut: bool = True):
        # Create new population either from parent or best
        self.population = [copy.deepcopy(parent) for _ in range(self.size)]

        # Mutate the new population
        if do_mut:
            self.mutate()
        
        # Reset all tensors, they are not valid in this new population
        self.areas = torch.empty(self.size, device=self.device)
        self.errors = torch.empty(self.size, device=self.device)
        self.fitnesses = torch.empty(self.size, device=self.device)

    def mutate(self, m_rate: float = None):
        # !!! Does not save the mutation rate
        if not m_rate:
            m_rate = self.m_rate

        # Mutate everyone except the first one (best)
        for individual in self.population[1:]:
            # Choose random nodes from active and mutate
            num_mutations = int(m_rate * len(individual))
            active_idxs = individual.get_active_idxs()
            mutation_idxs = torch.randperm(len(active_idxs), device=self.device)[:num_mutations]
            self.mutate_core(individual, mutation_idxs)
            
            # Choose random outputs and mutate
            num_mutations = int(m_rate * len(individual.outputs))
            mutation_idxs = torch.randperm(len(individual.outputs), device=self.device)[:num_mutations]
            self.mutate_outputs(individual, mutation_idxs)
    
    def mutate_core(self, individual: CGPCircuit, mutation_indices: torch.Tensor):
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
        for idx in mutation_indices:
            individual.outputs[idx] = torch.randint(self.c_in + len(individual.core), (1,)).item()

    def calc_fitnesses(self) -> float:
        # # Reset all fittness tensors
        # self.areas = torch.empty(self.size, device=self.device)
        # self.errors = torch.empty(self.size, device=self.device)
        # self.fitnesses = torch.empty(self.size, device=self.device)

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

        if self.criterion == "area" and self.best_error > self.tau:
            print("Parent:", self.errors[0].item(), self.get_error(self.population[0]))
            print("Best:", self.best_error, self.get_error(self.population[best_idx]))
            raise ValueError("Best individual has fitness higher than tau. This should not happen.")
        elif self.criterion == "error" and self.best_area > self.tau:
            print("Parent:", self.areas[0].item(), len(self.population[0]))
            print("Best:", self.best_area, len(self.population[best_idx]))
            raise ValueError("Best individual has area higher than tau. This should not happen.")

        return self.fitnesses
    
    def get_error(self, individual: CGPCircuit) -> float:
        in_bits = int(self.c_in / self.c_ni)
        total_bits = (2 ** self.c_in) * self.c_in

        cum_hamm = 0
        for in1, in2, ref in generator.generate_all_vec(in_bits, self.batch_size, device=self.device):
            batch_input = torch.cat((in1, in2), dim=1).to(self.device)
            batch_output = individual.forward_batch(batch_input, device=self.device)

            out_flat = batch_output.flatten()
            ref_flat = ref.flatten()

            cum_hamm += torch.sum(out_flat ^ ref_flat).item()

        avg_error = (cum_hamm / total_bits) * 100
        return avg_error

    def get_best(self) -> Individual:
        return Individual(self.best, self.best_area, self.best_error, self.best_fitness)


fitness_functions = {
    "area": lambda areas, errs, tau: torch.where(
        errs <= tau,
        areas,
        torch.full_like(areas, float("inf"))
    ),
    "error": lambda areas, errs, tau: torch.where(
        areas <= tau,
        errs,
        torch.full_like(errs, float("inf"))
    )
}
