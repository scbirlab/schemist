"""Tools for enumerating compounds. Currently only works with peptides."""

from typing import Callable, Iterable, Optional, Tuple, Union

from functools import partial
from itertools import chain, islice, product, repeat
from math import ceil, expm1, floor
from random import choice, choices, random, seed

from carabiner import print_err
from carabiner.decorators import vectorize, return_none_on_error
from carabiner.random import sample_iter
from rdkit.Chem import Mol, rdChemReactions
import numpy as np

from .converting import (_x2mol, _mol2x,
                         _convert_input_to_smiles)

AA = tuple('GALVITSMCPFYWHKRDENQ')
dAA = tuple(aa.casefold() for aa in AA)

REACTIONS = {'N_to_C_cyclization': '([N;H1:5][C:1][C:2](=[O:6])[O:3].[N;H2:4][C:7][C:8](=[O:9])[N;H1:10])>>[N;H1:5][C:1][C:2](=[O:6])[N;H1:4][C:7][C:8](=[O:9])[N;H1:10].[O;H2:3]',
             'cysteine_to_chloroacetyl_cyclization': '([N;H1:5][C:2](=[O:6])[C:1][Cl:3].[S;H1:4][C;H2:7][C:8])>>[N;H1:5][C:2](=[O:6])[C:1][S:4][C;H2:7][C:8]',
             'cysteine_to_N_cyclization':'([N;H1:5][C:2](=[O:6])[C:1][N;H2:3].[S;H1:4][C;H2:7][C:8])>>[N;H1:5][C:2](=[O:6])[C:1][S:4][C;H2:7][C:8].[N;H3:3]'}

def _get_alphabet(alphabet: Optional[Iterable[str]] = None,
                  d_aa_only: bool = False,
                  include_d_aa: bool = False) -> Tuple[str]:

    alphabet = alphabet or AA
    alphabet_lower = tuple(set(aa.casefold() for aa in AA))

    if d_aa_only:
        alphabet = alphabet_lower
    elif include_d_aa:
        alphabet = tuple(set(chain(alphabet, alphabet_lower)))

    return alphabet

    

def all_peptides_of_one_length(length: int,
                               alphabet: Optional[Iterable[str]] = None,
                               d_aa_only: bool = False,
                               include_d_aa: bool = False) -> Iterable[str]:
    
    """
    
    """
    
    alphabet = _get_alphabet(alphabet=alphabet,
                             d_aa_only=d_aa_only,
                             include_d_aa=include_d_aa)

    return (''.join(peptide)
            for peptide in product(alphabet, repeat=length))


def all_peptides_in_length_range(max_length: int,
                                 min_length: int = 1,
                                 by: int = 1,
                                 alphabet: Optional[Iterable[str]] = None,
                                 d_aa_only: bool = False,
                                 include_d_aa: bool = False,
                                 *args, **kwargs) -> Iterable[str]:

    """
    
    """
    
    length_range = range(*sorted([min_length, max_length + 1]), by)
    peptide_maker = partial(all_peptides_of_one_length, 
                            alphabet=alphabet,
                            d_aa_only=d_aa_only,
                            include_d_aa=include_d_aa,
                            *args, **kwargs)
    
    return chain.from_iterable(peptide_maker(length=length) 
                               for length in length_range)


def _number_of_peptides(max_length: int,
                        min_length: int = 1,
                        by: int = 1,
                        alphabet: Optional[Iterable[str]] = None,
                        d_aa_only: bool = False,
                        include_d_aa: bool = False):
    
    alphabet = _get_alphabet(alphabet=alphabet,
                             d_aa_only=d_aa_only,
                             include_d_aa=include_d_aa)
    n_peptides = [len(alphabet) ** length 
                  for length in range(*sorted([min_length, max_length + 1]), by)]
    
    return n_peptides


def _naive_sample_peptides_in_length_range(max_length: int,
                                           min_length: int = 1,
                                           by: int = 1,
                                           n: Optional[Union[float, int]] = None,
                                           alphabet: Optional[Iterable[str]] = None,
                                           d_aa_only: bool = False,
                                           include_d_aa: bool = False,
                                           set_seed: Optional[int] = None):
    
    alphabet = _get_alphabet(alphabet=alphabet,
                             d_aa_only=d_aa_only,
                             include_d_aa=include_d_aa)
    n_peptides = _number_of_peptides(max_length=max_length,
                                     min_length=min_length,
                                     by=by,
                                     alphabet=alphabet,
                                     d_aa_only=d_aa_only,
                                     include_d_aa=include_d_aa)
    lengths = list(range(*sorted([min_length, max_length + 1]), by))
    weight_per_length = [n / min(n_peptides) for n in n_peptides]
    weighted_lengths = list(chain.from_iterable(repeat(l, ceil(w)) for l, w in zip(lengths, weight_per_length)))

    lengths_sample = (choice(weighted_lengths) for _ in range(n))
    return (''.join(choices(list(alphabet), k=k)) for k in lengths_sample)


def sample_peptides_in_length_range(max_length: int,
                                    min_length: int = 1,
                                    by: int = 1,
                                    n: Optional[Union[float, int]] = None,
                                    alphabet: Optional[Iterable[str]] = None,
                                    d_aa_only: bool = False,
                                    include_d_aa: bool = False,
                                    naive_sampling_cutoff: float = 5e-3,
                                    reservoir_sampling: bool = True,
                                    indexes: Optional[Iterable[int]] = None,
                                    set_seed: Optional[int] = None,
                                    *args, **kwargs) -> Iterable[str]:

    """

    """

    seed(set_seed)

    alphabet = _get_alphabet(alphabet=alphabet,
                             d_aa_only=d_aa_only,
                             include_d_aa=include_d_aa)

    n_peptides = sum(len(alphabet) ** length 
                     for length in range(*sorted([min_length, max_length + 1]), by))
    if n is None:
        n_requested = n_peptides
    elif n >= 1.:
        n_requested = min(floor(n), n_peptides)
    elif n < 1.:
        n_requested = floor(n * n_peptides)

    frac_requested = n_requested / n_peptides
    
    # approximation of birthday problem
    p_any_collision = -expm1(-n_requested * (n_requested - 1.) / (2. * n_peptides))
    n_collisons = n_requested * (1. - ((n_peptides - 1.) / n_peptides) ** (n_requested - 1.))
    frac_collisions = n_collisons / n_requested

    print_err(f"Sampling {n_requested} ({frac_requested * 100.} %) peptides from "
              f"length {min_length} to {max_length} ({n_peptides} combinations). "
              f"Probability of collision if drawing randomly is {p_any_collision}, "
              f"with {n_collisons} ({100. * frac_collisions} %) collisions on average.")

    if frac_collisions < naive_sampling_cutoff and n_peptides > 2e9:

        print_err("> Executing naive sampling. ")

        peptides = _naive_sample_peptides_in_length_range(max_length, min_length, by, 
                                                          n=n_requested,
                                                          alphabet=alphabet,
                                                          d_aa_only=d_aa_only,
                                                          include_d_aa=include_d_aa)

    else:

        print_err("> Executing exhaustive sampling.")

        all_peptides = all_peptides_in_length_range(max_length, min_length, by, 
                                                    alphabet=alphabet,
                                                    d_aa_only=d_aa_only,
                                                    include_d_aa=include_d_aa,
                                                    *args, **kwargs)

        if n is None:

            peptides = all_peptides
            
        elif n >= 1.:
            
            if reservoir_sampling:
                peptides = sample_iter(all_peptides, k=n_requested,
                                       shuffle_output=False)
            else:
                peptides = (pep for pep in all_peptides 
                            if random() <= frac_requested)

        elif n < 1.:

            peptides = (pep for pep in all_peptides 
                        if random() <= n)

    if indexes is not None:

        indexes = (int(ix) if (isinstance(ix, str) and ix.isdigit()) or isinstance(ix, int) or isinstance(ix, float) 
                   else None 
                   for ix in islice(indexes, 3))
        indexes = [ix if (ix is None or ix >= 0) else None 
                   for ix in indexes]
        
        if len(indexes) > 1:
            if n is not None and n >=1. and indexes[0] > n:
                raise ValueError(f"Minimum slice ({indexes[0]}) is higher than number of items ({n}).")

        peptides = islice(peptides, *indexes)

    return peptides


def _reactor(smarts: str) -> Callable[[Mol], Union[Mol, None]]:

    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    reaction_function = rxn.RunReactants

    @vectorize
    @return_none_on_error
    def reactor(s: Mol) -> Mol:

        return reaction_function([s])[0][0]
    
    return reactor


@_convert_input_to_smiles
def react(strings: Union[str, Iterable[str]], 
          reaction: str = 'N_to_C_cyclization', 
          output_representation: str = 'smiles',
          **kwargs) -> Union[str, Iterable[str]]:
    
    """
    
    """

    try:
        _this_reaction = REACTIONS[reaction]
    except KeyError:
        raise KeyError(f"Reaction {reaction} is not available. Try: " +
                        ", ".join(list(REACTIONS)))

    # strings = cast(strings, to=list)
    # print_err((strings))

    reactor = _reactor(_this_reaction)
    mols = _x2mol(strings)
    mols = reactor(mols)

    return _mol2x(mols, 
                  output_representation=output_representation,
                  **kwargs)
