import torch

#############################
# Vectorized implementation #
#############################

def uint_to_binary_tensor_vec(nums: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Converts a tensor of unsigned integers to a binary tensor representation.
    The tensor is expected to be of shape (N,) where N is the number of integers.
    The resulting tensor will have shape (N, bits) where each row is the binary representation of the corresponding
    integer.

    :param nums: A tensor of unsigned integers
    :param bits: The number of bits to represent each integer
    :return: A binary tensor representation of the input tensor
    """
    # !!! There are no checks for a valid input
    shifts = torch.arange(0, bits, device=nums.device)
    return ((nums.unsqueeze(1) >> shifts) & 1).bool()

#################################
# Non-vectorized implementation #
#################################

def uint_to_binary_tensor(num: int, bits: int) -> torch.Tensor:
    """
    Converts an unsigned integer to a binary tensor representation.

    :param num: The unsigned integer
    :param bits: The number of bits to represent the integer
    :return: A binary tensor representation of the input integer
    """
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
    """"
    Converts a binary tensor to an unsigned integer.
    
    :param tensor: A binary tensor
    :return: The unsigned integer representation of the input tensor
    """
    if tensor.dtype != torch.bool:
        raise ValueError("Input tensor must be of type bool")
    
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")
    
    n = tensor.shape[0]
    weights = 2 ** torch.arange(n, device=tensor.device)
    return torch.sum(tensor * weights).item()