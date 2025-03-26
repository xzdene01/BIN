"""
@file   mappings.py
@brief  Contains mappings of opcodes (INT) to their string representation and logical function.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import torch

# Mapping of operation codes to their string representation
opcode_to_str = {
    0: "IDENTITY",
    1: "NOT",
    2: "AND",
    3: "OR",
    4: "XOR",
    5: "NAND",
    6: "NOR",
    7: "XNOR",
    8: "TRUE",
    9: "FALSE"
}

# Mapping of operation codes to their logical function
opcode_to_func = {
    0: lambda a, b: a,                                                      # IDENTITY: returns the first input
    1: lambda a, b: torch.logical_not(a),                                   # NOT: ignores b
    2: lambda a, b: torch.logical_and(a, b),                                # AND
    3: lambda a, b: torch.logical_or(a, b),                                 # OR
    4: lambda a, b: torch.logical_xor(a, b),                                # XOR
    5: lambda a, b: torch.logical_not(torch.logical_and(a, b)),             # NAND
    6: lambda a, b: torch.logical_not(torch.logical_or(a, b)),              # NOR
    7: lambda a, b: torch.logical_not(torch.logical_xor(a, b)),             # XNOR
    8: lambda a, b: torch.tensor(True, dtype=torch.bool, device=a.device),  # TRUE constant
    9: lambda a, b: torch.tensor(False, dtype=torch.bool, device=a.device)  # FALSE constant
}