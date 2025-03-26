"""
@file   mappings.py
@brief  Mapping of fitness functions to their implementations.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import torch

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