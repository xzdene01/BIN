"""
@file   maybe_tqdm.py
@brief  This file contains a function that returns a tqdm object if the use_tqdm parameter is True,
        otherwise it returns just the iterable object itself.
@author Jan Zdeněk (xzdene01)
@date   26/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import tqdm

def maybe_tqdm(iterable, use_tqdm: bool, **kwargs):
    if use_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    return iterable