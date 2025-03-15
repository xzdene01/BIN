import tqdm

def maybe_tqdm(iterable, use_tqdm: bool, **kwargs):
    if use_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    return iterable