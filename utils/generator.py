"""
@file   generator.py
@brief  Functions for generating all possible combinations of inputs and outputs.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import torch
import tqdm

from utils import convert

#############################
# Vectorized implementation #
#############################

def generate_all_vec(bits: int, chunk_size: int, device: str = "cpu"):
    """
    Generates all possible combinations of inputs and outputs for the multiplication operation.
    The inputs and outputs are represented as binary tensors.
    
    :param bits: The number of bits to represent each input
    :param chunk_size: The size of the chunk for vectorized operations
    :param device: The device to use for the tensors
    :yield: A tuple of input1, input2, and output binary tensors
    """
    in1_chunk_size = max(1, chunk_size // (2 ** bits))

    # This is a bottleneck for large bit sizes
    in_range = torch.arange(0, 2**bits, device=device)

    total = 2 ** bits
    for start in range(0, total, in1_chunk_size):
        end = min(start + in1_chunk_size, total)
        chunk_in1 = in_range[start:end]

        # Get all possible combinations of in1 and in2, where in1 will be chunked
        in1_grid, in2_grid = torch.meshgrid(chunk_in1, in_range, indexing='ij')
        in1_flat = in1_grid.flatten()
        in2_flat = in2_grid.flatten()

        # Cumpute the output for all combinations if inputs
        out_flat = in1_flat * in2_flat

        in1_tensor = convert.uint_to_binary_tensor_vec(in1_flat, bits)
        in2_tensor = convert.uint_to_binary_tensor_vec(in2_flat, bits)
        out_tensor = convert.uint_to_binary_tensor_vec(out_flat, 2 * bits)

        yield in1_tensor, in2_tensor, out_tensor

#################################
# Non-vectorized implementation #
#################################

def generate_all(bits: int, device: str = "cpu") -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    triplets = []
    for input1 in tqdm.tqdm(range(2 ** bits)):
        for input2 in range(2 ** bits):
            output = input1 * input2
            input1_tensor = convert.uint_to_binary_tensor(input1, bits).to(device)
            input2_tensor = convert.uint_to_binary_tensor(input2, bits).to(device)
            output_tensor = convert.uint_to_binary_tensor(output, 2 * bits).to(device)
            triplets.append((input1_tensor, input2_tensor, output_tensor))
    return triplets

def save_triplets(triplets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], file: str):
    torch.save(triplets, file)

def load_triplets(file: str, device: str = "cpu") -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return torch.load(file, map_location=device)