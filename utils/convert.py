"""
@file   convert.py
@brief  Conversion between numbers and binary tensors
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

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
    shifts = torch.arange(0, bits, device=nums.device)
    return ((nums.unsqueeze(1) >> shifts) & 1).bool()

def binary_tensor_to_uint_vec(tensor: torch.Tensor) -> torch.Tensor:
    """"
    Converts a binary tensor to an unsigned integer.
    The tensor is expected to be of shape (N, bits) where N is the number of integers.
    The resulting tensor will have shape (N,) where each element is the unsigned integer representation of the
    corresponding row in the input tensor.

    :param tensor: A binary tensors
    :return: The unsigned integer representation of the input tensors
    """
    num_bits = tensor.shape[1]
    weights = 2 ** torch.arange(num_bits - 1, -1, -1, device=tensor.device, dtype=torch.long)
    return (tensor.long() * weights).sum(dim=1)

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
    n = tensor.shape[0]
    weights = 2 ** torch.arange(n, device=tensor.device)
    return torch.sum(tensor * weights).item()