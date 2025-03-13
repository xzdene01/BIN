import torch
import tqdm

from utils import convert

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

def generate_all_vec(bits: int, chunk_size: int, device: str = "cuda"):
    chunk_size = max(1, chunk_size // (2 ** bits))
    in_range = torch.arange(0, 2**bits, device=device)
    total = 2**bits

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_in1 = in_range[start:end]

        in1_grid, in2_grid = torch.meshgrid(chunk_in1, in_range, indexing='ij')

        in1_flat = in1_grid.flatten()
        in2_flat = in2_grid.flatten()

        out_flat = in1_flat * in2_flat

        in1_bin = convert.uint_to_binary_tensor_vec(in1_flat, bits)
        in2_bin = convert.uint_to_binary_tensor_vec(in2_flat, bits)
        out_bin = convert.uint_to_binary_tensor_vec(out_flat, 2 * bits)

        yield in1_bin, in2_bin, out_bin

# def generate_all_vectorized(bits: int, device: str = "cuda"):
#     # Generate a range of N-bit values and create all (in1, in2) pairs using meshgrid.
#     in_range = torch.arange(0, 2**bits, device=device)
#     in1_grid, in2_grid = torch.meshgrid(in_range, in_range, indexing='ij')
    
#     # Flatten the grids to get 1D lists of inputs.
#     in1_flat = in1_grid.flatten()
#     in2_flat = in2_grid.flatten()
    
#     # Multiply them elementwise to get the product for each pair.
#     out_flat = in1_flat * in2_flat
    
#     # Convert everything to binary tensors.
#     in1_bin = convert.uint_to_binary_tensor_vec(in1_flat, bits)
#     in2_bin = convert.uint_to_binary_tensor_vec(in2_flat, bits)
#     out_bin = convert.uint_to_binary_tensor_vec(out_flat, 2 * bits)

#     return in1_bin, in2_bin, out_bin

def save_triplets(triplets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], file: str):
    torch.save(triplets, file)

def load_triplets(file: str, device: str = "cpu") -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return torch.load(file, map_location=device)