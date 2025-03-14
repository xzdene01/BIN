import torch
import tqdm
import copy

from circuit.cgp_circuit import CGPCircuit, opcode_to_str
from utils import generator


class Individual:
    def __init__(self, c_in, core, outputs):
        self.c_in = c_in
        self.core = core
        self.outputs = outputs

        self.active_mask = CGPCircuit.get_active_mask_static(c_in, core, outputs)
        self.active_idxs = torch.nonzero(self.active_mask).flatten()
        self.len = int(self.active_mask.sum().item())
    
    def mutate_core(self, mutation_indices: torch.Tensor):
        for idx in mutation_indices:
            rnd_num = torch.rand(1).item()
            if rnd_num < 0.3:
                self.core[idx].op_code = torch.randint(len(opcode_to_str), (1,)).item()
            elif rnd_num < 0.65:
                self.core[idx].in_1 = torch.randint(self.c_in + len(self.core), (1,)).item()
            else:
                self.core[idx].in_2 = torch.randint(self.c_in + len(self.core), (1,)).item()
    
    def mutate_outputs(self, mutation_indices: torch.Tensor):
        pass
    
    def recalculate(self):
        self.active_mask = CGPCircuit.get_active_mask_static(self.c_in, self.core, self.outputs)
        self.active_idxs = torch.nonzero(self.active_mask).flatten()
        self.len = int(self.active_mask.sum().item())

    def __len__(self):
        return self.len


# TODO change to use CGPCircuit as individual
class Population:
    def __init__(self, size: int, m_rate: float, criterion: str, cgp: CGPCircuit, mutate: bool = False):
        self.size = size
        self.m_rate = m_rate
        self.criterion = criterion

        self.prefix = cgp.prefix
        self.best = Individual(self.prefix["c_in"], cgp.core, cgp.outputs)
        self.best_area = None
        self.best_error = None

        self.population = []
        self.populate(self.best, mutate=mutate)

        self.areas = torch.empty(size)
        self.errors = torch.empty(size)
        self.fitnesses = torch.empty(size)

    def populate(self, parent: Individual, mutate: bool = True):
        self.population = []
        for i in range(self.size):
            self.population.append(copy.deepcopy(parent))

        if mutate:
            self.mutate()
        
        # Reset all tensors, they are not valid in the new population
        self.areas = torch.empty(self.size)
        self.errors = torch.empty(self.size)
        self.fitnesses = torch.empty(self.size)

    def mutate(self, m_rate: float = None):
        # !!! Does not save the mutation rate
        if not m_rate:
            m_rate = self.m_rate

        # Mutate everyone except the first one (parent)
        for individual in self.population[1:]:
            # Choose random active nodes
            num_mutations = int(m_rate * len(individual))
            active_idxs = individual.active_idxs
            mutation_idxs = torch.randperm(len(active_idxs))[:num_mutations]
            individual.mutate_core(mutation_idxs)
            
            # Choose random outputs
            num_mutations = int(m_rate * len(individual.outputs))
            mutation_idxs = torch.randperm(len(individual.outputs))[:num_mutations]
            individual.mutate_outputs(mutation_idxs)
            
            # Recalculate active nodes
            individual.recalculate()

    def get_fittness(self, batch_size, device) -> float:
        c_in = self.prefix["c_in"]
        c_ni = self.prefix["c_ni"]
        in_bits = int(c_in / c_ni)

        # Reset all tensors
        self.areas = torch.empty(self.size)
        self.errors = torch.empty(self.size)
        self.fitnesses = torch.empty(self.size)

        for i, individual in enumerate(tqdm.tqdm(self.population, desc="Calculating fitness", unit="individual")):
            cum_hamm = 0
            cum_total = 0
            for in1, in2, ref in generator.generate_all_vec(in_bits, batch_size, device=device):
                batch_input = torch.cat((in1, in2), dim=1)
                batch_output = CGPCircuit.forward_static_batch(batch_input, self.prefix["c_in"], individual.core, individual.outputs).to(device)

                out_flat = batch_output.flatten()
                ref_flat = ref.flatten()

                cum_hamm += torch.sum(out_flat ^ ref_flat).item()
                cum_total += ref_flat.size(0)

            total_bits = (2 ** c_in) * c_in
            avg_error = cum_hamm / total_bits * 100
            self.areas[i] = len(individual)
            self.errors[i] = avg_error

        # !!! Hardcoded tau
        tau = 10
        self.fitnesses = fitness_functions[self.criterion](self.areas, self.errors, tau)

        # Current best must always be at index 0
        # TODO if more with same fittness choose best on the second criterion
        idx, val = min(enumerate(self.fitnesses), key=lambda pair: (pair[1], pair[0]))
        self.best = self.population[idx]
        self.best_area = int(self.areas[idx].item())
        self.best_error = self.errors[idx].item()

        if idx == 0:
            print("Best has not changed")

        # print("Areas:", self.areas)
        # print("Errors:", self.errors)
        # print("Fitnesses:", self.fitnesses)

        return self.fitnesses

    def get_best(self) -> CGPCircuit:
        return self.best, self.best_area, self.best_error


fitness_functions = {
    "area": lambda areas, errs, tau: torch.where(
        errs <= tau,
        areas,
        torch.full_like(areas, float('inf'))
    ),
    "error": lambda areas, errs, tau: torch.where(
        areas <= tau,
        errs,
        torch.full_like(errs, float('inf'))
    )
}
