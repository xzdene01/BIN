import torch
import tqdm

from circuit.cgp_circuit import CGPCircuit
from utils import generator

class Individual:
    def __init__(self, core, outputs):
        self.core = core
        self.outputs = outputs

class Population:
    def __init__(self, size: int, m_rate: float, cgp: CGPCircuit):
        self.size = size
        self.m_rate = m_rate
        self.prefix = cgp.prefix
        self.best = Individual(cgp.core, cgp.outputs)

        self.population = []
        self.populate(cgp, muatate=True)

        self.fitnesses = []
    
    def populate(self, cgp: CGPCircuit, muatate: bool = True):
        self.population = [individual for individual in [Individual(cgp.core, cgp.outputs) for _ in range(self.size)]]
        if muatate:
            self.mutate()
    
    def mutate(self, m_rate: float = None):
        # !!! Does not save the mutation rate
        if not m_rate:
            m_rate = self.m_rate
        pass

    def get_fittness(self, batch_size, device) -> float:
        in_bits = int(self.prefix["c_in"] / self.prefix["c_ni"])
        total_trials = int(2 ** (2 * in_bits))

        self.fitnesses = []
        for individual in tqdm.tqdm(self.population, desc="Calculating fitness", unit="individual"):
            succ = 0
            for in1_bin, in2_bin, out_bin in generator.generate_all_vec(in_bits, batch_size, device=device):
                batch_input = torch.cat((in1_bin, in2_bin), dim=1)
                batch_output = CGPCircuit.forward_static_batch(batch_input, self.prefix["c_in"], individual.core, individual.outputs).to(device)

                succ += (batch_output == out_bin).all(dim=1).sum().item()
            
            self.fitnesses.append(succ / total_trials)
        
        # Current best must always be at index 0
        curr_fitness = self.fitnesses[0]
        new_fitnesses = self.fitnesses[1:]
        if max(new_fitnesses) >= curr_fitness:
            self.best = self.population[new_fitnesses.index(max(new_fitnesses)) + 1]

    def get_best(self) -> CGPCircuit:
        cgp = CGPCircuit()
        cgp.prefix = self.prefix
        cgp.core = self.best.core
        cgp.outputs = self.best.outputs
        return cgp
    