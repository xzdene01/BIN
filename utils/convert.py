import torch

def uint_to_binary_tensor(num: int, bits: int) -> torch.Tensor:
    if num < 0:
        raise ValueError("Number must be greater than or equal to 0")

    if bits < 1:
        raise ValueError("Number of bits must be greater than 0")

    int_str = f"{num:0{bits}b}"

    if len(int_str) > bits:
        raise ValueError(f"Number {num} cannot be represented with {bits} bits")

    bit_list = [int(bit) for bit in int_str]
    bit_list = bit_list[::-1]
    return torch.tensor(bit_list, dtype=torch.bool)

def binary_tensor_to_uint(tensor: torch.Tensor) -> int:
    if tensor.dtype != torch.bool:
        raise ValueError("Input tensor must be of type bool")
    
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")
    
    num = 0
    for i, bit in enumerate(tensor):
        if bit:
            num += 2 ** i

    return num

def uint_to_binary_tensor_vec(nums: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(0, bits, device=nums.device)
    return ((nums.unsqueeze(1) >> shifts) & 1).bool()