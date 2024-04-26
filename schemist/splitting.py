"""Tools for splitting tabular datasets, optionally based on chemical features."""

from typing import Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from math import ceil
from random import random, seed

try:
    from itertools import batched
except ImportError:
    from carabiner.itertools import batched

from tqdm.auto import tqdm

from .converting import convert_string_representation, _convert_input_to_smiles
from .typing import DataSplits

# def _train_test_splits

def _train_test_val_sizes(total: int, 
                          train: float = 1., 
                          test: float = 0.) -> Tuple[int]:
    
    n_train = int(ceil(train * total))
    n_test = int(ceil(test * total))
    n_val = total - n_train - n_test

    return n_train, n_test, n_val
    

def _random_chunk(strings: str,
                  train: float = 1., 
                  test: float = 0.,
                  carry: Optional[Dict[str, List[int]]] = None,
                  start_from: int = 0) -> Dict[str, List[int]]:
    
    carry = carry or defaultdict(list)

    train_test: float = train + test

    for i, _ in enumerate(strings):

        random_number: float = random()

        if random_number < train:

            key = 'train'

        elif random_number < train_test:

            key = 'test'

        else:

            key = 'validation'

        carry[key].append(start_from + i)

    return carry


def split_random(strings: Union[str, Iterable[str]], 
                 train: float = 1., 
                 test: float = 0.,
                 chunksize: Optional[int] = None,
                 set_seed: Optional[int] = None,
                 *args, **kwargs) -> DataSplits:
    
    """
    
    """

    if set_seed is not None:

        seed(set_seed)


    if chunksize is None:

        idx = _random_chunk(strings=strings, 
                            train=train, 
                            test=test)

    else:

        idx = defaultdict(list)

        for i, chunk in enumerate(batched(strings, chunksize)):

            idx = _random_chunk(strings=chunk, 
                                train=train, 
                                test=test,
                                carry=idx, 
                                start_from=i * chunksize)
            
    seed(None)
    
    return DataSplits(**idx)
    

@_convert_input_to_smiles
def _scaffold_chunk(strings: str,
                    carry: Optional[Dict[str, List[int]]] = None,
                    start_from: int = 0) -> Dict[str, List[int]]:
    
    carry = carry or defaultdict(list)
    
    these_scaffolds = convert_string_representation(strings=strings,
                                                    output_representation='scaffold')
    
    for j, scaff in enumerate(these_scaffolds):
        carry[scaff].append(start_from + j)

    return carry


def _scaffold_aggregator(scaffold_sets: Dict[str, List[int]],
                         train: float = 1., 
                         test: float = 0.,
                         progress: bool = False) -> DataSplits:

    scaffold_sets = {key: sorted(value) 
                     for key, value in scaffold_sets.items()}
    scaffold_sets = sorted(scaffold_sets.items(),
                           key=lambda x: (len(x[1]), x[1][0]),
                           reverse=True)
    nrows = sum(len(idx) for _, idx in scaffold_sets)
    n_train, n_test, n_val = _train_test_val_sizes(nrows,
                                                   train, 
                                                   test)
    idx = defaultdict(list)

    iterator = tqdm(scaffold_sets) if progress else scaffold_sets
    for _, scaffold_idx in iterator:

        if (len(idx['train']) + len(scaffold_idx)) > n_train:

            if (len(idx['test']) + len(scaffold_idx)) > n_test:

                key = 'validation'

            else:

                key = 'test'
        else:

            key = 'train' 

        idx[key] += scaffold_idx

    return DataSplits(**idx)


def split_scaffold(strings: Union[str, Iterable[str]], 
                   train: float = 1., 
                   test: float = 0.,
                   chunksize: Optional[int] = None, 
                   progress: bool = True,
                   *args, **kwargs) -> DataSplits:

    """
    
    """

    if chunksize is None:

        scaffold_sets = _scaffold_chunk(strings)
    
    else:

        scaffold_sets = defaultdict(list)

        for i, chunk in enumerate(batched(strings, chunksize)):

            scaffold_sets = _scaffold_chunk(chunk, 
                                            carry=scaffold_sets, 
                                            start_from=i * chunksize)

    return _scaffold_aggregator(scaffold_sets, 
                                train=train, test=test, 
                                progress=progress)


_SPLITTERS = {#'simpd': split_simpd, 
              'scaffold': split_scaffold, 
              'random': split_random}

# _SPLIT_SUPERTYPES = {'scaffold': 'grouped', 
#                      'random': 'independent'}

_GROUPED_SPLITTERS = {'scaffold': (_scaffold_chunk, _scaffold_aggregator)}

assert all(_type in _SPLITTERS 
           for _type in _GROUPED_SPLITTERS)  ## Should never fail!

def split(split_type: str,
          *args, **kwargs) -> DataSplits:
    
    """
    
    """
    
    splitter = _SPLITTERS[split_type]

    return splitter(*args, **kwargs)