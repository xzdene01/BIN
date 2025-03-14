import torch
import tqdm

from circuit.cgp_circuit import CGPCircuit
from utils import generator


class Individual:
    def __init__(self, c_in, core, outputs):
        self.c_in = c_in
        self.core = core
        self.outputs = outputs

    def __len__(self):
        mask = CGPCircuit.get_active_mask(self.c_in, self.core, self.outputs)
        return int(mask.sum().item())


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
        self.populate(cgp, mutate=mutate)

        self.areas = torch.empty(size)
        self.errors = torch.empty(size)
        self.fitnesses = torch.empty(size)

    def populate(self, cgp: CGPCircuit, mutate: bool = True):
        self.population = [individual for individual in [Individual(self.prefix["c_in"], cgp.core, cgp.outputs) for _ in range(self.size)]]
        if mutate:
            self.mutate()

    def mutate(self, m_rate: float = None):
        # !!! Does not save the mutation rate
        if not m_rate:
            m_rate = self.m_rate
        pass

    def get_fittness(self, batch_size, device) -> float:
        in_bits = int(self.prefix["c_in"] / self.prefix["c_ni"])

        # Reset all tensors
        self.areas = torch.empty(self.size)
        self.errors = torch.empty(self.size)
        self.fitnesses = torch.empty(self.size)

        for i, individual in enumerate(tqdm.tqdm(self.population, desc="Calculating fitness", unit="individual")):
            cum_hamm = 0
            total_bits = 2 ** self.prefix["c_in"]
            for in1_bin, in2_bin, out_bin in generator.generate_all_vec(in_bits, batch_size, device=device):
                batch_input = torch.cat((in1_bin, in2_bin), dim=1)
                batch_output = CGPCircuit.forward_static_batch(batch_input, self.prefix["c_in"], individual.core, individual.outputs).to(device)

                cum_hamm += torch.sum(batch_output ^ out_bin)

            avg_error = cum_hamm / total_bits
            self.areas[i] = len(individual)
            self.errors[i] = avg_error

        self.fitnesses = fitness_functions[self.criterion](self.areas, self.errors, 0.1)

        # Current best must always be at index 0
        idx, val = max(enumerate(self.fitnesses), key=lambda pair: (pair[1], pair[0]))
        self.best = self.population[idx]
        self.best_area = int(self.areas[idx].item())
        self.best_error = self.errors[idx].item()

        return self.fitnesses

    def get_best(self) -> CGPCircuit:
        cgp = CGPCircuit()
        cgp.prefix = self.prefix
        cgp.core = self.best.core
        cgp.outputs = self.best.outputs
        return cgp, self.best_area, self.best_error


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
