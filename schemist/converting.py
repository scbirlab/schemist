"""Converting between chemical representation formats."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from functools import wraps

from carabiner import print_err
from carabiner.cast import cast, flatten
from carabiner.decorators import return_none_on_error, vectorize
from carabiner.itertools import batched

# from datamol import sanitize_smiles
import nemony as nm
from pandas import DataFrame
from rdkit.Chem import (
    Crippen, 
    Descriptors, 
    rdMolDescriptors,
    Mol, 
    MolFromInchi, 
    MolFromHELM, 
    MolFromSequence, 
    MolFromSmiles, 
    MolToInchi, 
    MolToInchiKey, 
    MolToSmiles,
    SanitizeFlags,
    SanitizeMol
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from requests import Session
import selfies as sf

from .cleaning import sanitize_smiles_to_mol
from .rest_lookup import _inchikey2pubchem_name_id, _inchikey2cactus_name

@vectorize
@return_none_on_error
def _seq2mol(s: str) -> Union[Mol, None]:

    return MolFromSequence(s, sanitize=True)


@vectorize
@return_none_on_error
def _helm2mol(s: str) -> Union[Mol, None]:

    return MolFromHELM(s, sanitize=True)


def mini_helm2helm(s: str) -> List[str]:

    new_s = []
    token = ''
    between_sq_brackets = False

    for letter in s:

        if letter.islower() and not between_sq_brackets:

            letter = f"[d{letter.upper()}]"

        token += letter

        if letter == '[':
            between_sq_brackets = True
        elif letter == ']':
            between_sq_brackets = False

        if not between_sq_brackets:
            new_s.append(token)
            token = ''

    return "PEPTIDE1{{{inner_helm}}}$$$$".format(inner_helm='.'.join(new_s))


@vectorize  
@return_none_on_error
def _mini_helm2mol(s: str) -> Mol:

    s = mini_helm2helm(s)

    return MolFromHELM(s, sanitize=True)


@vectorize
@return_none_on_error
def _inchi2mol(s: str) -> Mol:

    return MolFromInchi(s, 
                        sanitize=True, 
                        removeHs=True)


# @vectorize
# @return_none_on_error
# def _smiles2mol(s: str) -> Mol:

#     return sanitize_smiles_to_mol(s)
_smiles2mol = vectorize(return_none_on_error(sanitize_smiles_to_mol))

@vectorize
@return_none_on_error
def _selfies2mol(s: str) -> Mol:

    return sanitize_smiles_to_mol(sf.decoder(s))


@vectorize
@return_none_on_error
def _mol2clogp(m: Mol,
               **kwargs) -> float:

    return Crippen.MolLogP(m)


@vectorize
@return_none_on_error
def _mol2nonstandard_inchikey(m: Mol,
                              **kwargs) -> str:

    return MolToInchiKey(m, 
                         options="/FixedH /SUU /RecMet /KET /15T")


@vectorize
@return_none_on_error
def _mol2hash(m: Mol,
              **kwargs) -> str:

    nonstandard_inchikey = _mol2nonstandard_inchikey(m)

    return nm.hash(nonstandard_inchikey)


@vectorize
@return_none_on_error
def _mol2id(m: Mol, 
            n: int = 8,
            prefix: str = '',
            **kwargs) -> str:

    return prefix + str(int(_mol2hash(m), 16))[:n]


@vectorize
@return_none_on_error
def _mol2isomeric_canonical_smiles(m: Mol,
                                   **kwargs) -> str:

    return MolToSmiles(m,
                       isomericSmiles=True,
                       canonical=True)


@vectorize
@return_none_on_error
def _mol2inchi(m: Mol,
               **kwargs) -> str:

    return MolToInchi(m)


@vectorize
@return_none_on_error
def _mol2inchikey(m: Mol,
                  **kwargs) -> str:

    return MolToInchiKey(m)


@vectorize
@return_none_on_error
def _mol2random_smiles(m: Mol,
                       **kwargs) -> str:

    return MolToSmiles(m,
                       isomericSmiles=True,
                       doRandom=True)


@vectorize
@return_none_on_error
def _mol2mnemonic(m: Mol,
                  **kwargs) -> str:

    nonstandard_inchikey = _mol2nonstandard_inchikey(m)

    return nm.encode(nonstandard_inchikey)


@vectorize
@return_none_on_error
def _mol2mwt(m: Mol,
             **kwargs) -> float:

    return Descriptors.ExactMolWt(m)


@vectorize
@return_none_on_error
def _mol2min_charge(m: Mol,
                    **kwargs) -> float:

    return Descriptors.MinPartialCharge(m)


@vectorize
@return_none_on_error
def _mol2max_charge(m: Mol,
                    **kwargs) -> float:

    return Descriptors.MaxPartialCharge(m)


@vectorize
@return_none_on_error
def _mol2tpsa(m: Mol,
              **kwargs) -> float:

    return rdMolDescriptors.CalcTPSA(m)


def _mol2pubchem(m: Union[Mol, Iterable[Mol]],
                 session: Optional[Session] = None,
                 chunksize: int = 32) -> List[Dict[str, Union[None, int, str]]]:
    
    inchikeys = cast(_mol2inchikey(m), to=list)
    pubchem_ids = []

    for _inchikeys in batched(inchikeys, chunksize):

        these_ids = _inchikey2pubchem_name_id(_inchikeys, 
                                              session=session)
        pubchem_ids += these_ids

    return pubchem_ids


@return_none_on_error
def _mol2pubchem_id(m: Union[Mol, Iterable[Mol]],
                    session: Optional[Session] = None,
                    chunksize: int = 32,
                    **kwargs) -> Union[str, List[str]]:

    return flatten([val['pubchem_id'] 
                     for val in _mol2pubchem(m, 
                                             session=session, 
                                             chunksize=chunksize)])


@return_none_on_error
def _mol2pubchem_name(m: Union[Mol, Iterable[Mol]],
                      session: Optional[Session] = None,
                      chunksize: int = 32,
                      **kwargs) -> Union[str, List[str]]:

    return flatten([val['pubchem_name'] 
                     for val in _mol2pubchem(m, 
                                             session=session, 
                                             chunksize=chunksize)])

@return_none_on_error
def _mol2cactus_name(m: Union[Mol, Iterable[Mol]],
                     session: Optional[Session] = None,
                     **kwargs) -> Union[str, List[str]]:

    return _inchikey2cactus_name(_mol2inchikey(m), 
                                 session=session)


@vectorize
@return_none_on_error
def _mol2scaffold(m: Mol,
                  chiral: bool = True,
                  **kwargs) -> str:
    
    return MurckoScaffoldSmiles(mol=m, 
                                includeChirality=chiral)


@vectorize
@return_none_on_error
def _mol2selfies(m: Mol,
                 **kwargs) -> str:

    s = sf.encoder(_mol2isomeric_canonical_smiles(m))

    return s if s != -1 else None


_TO_FUNCTIONS = {"smiles": _mol2isomeric_canonical_smiles,
                 "selfies": _mol2selfies,
                 "inchi": _mol2inchi,
                 "inchikey": _mol2inchikey,
                 "nonstandard_inchikey": _mol2nonstandard_inchikey,
                 "hash": _mol2hash,
                 "mnemonic": _mol2mnemonic, 
                 "id": _mol2id,
                 "scaffold": _mol2scaffold,
                 "permuted_smiles": _mol2random_smiles,
                 "pubchem_id": _mol2pubchem_id,
                 "pubchem_name": _mol2pubchem_name, 
                 "cactus_name": _mol2cactus_name,
                 "clogp": _mol2clogp,
                 "tpsa": _mol2tpsa,
                 "mwt": _mol2mwt,
                 "min_charge": _mol2min_charge,
                 "max_charge": _mol2max_charge}

_FROM_FUNCTIONS = {"smiles": _smiles2mol,
                   "selfies": _selfies2mol,
                   "inchi": _inchi2mol,
                   "aa_seq": _seq2mol,
                   "helm": _helm2mol,
                   "minihelm": _mini_helm2mol}


def _x2mol(
    strings: Union[Iterable[str], str],
    input_representation: str = 'smiles'
) -> Union[Mol, None, Iterable[Union[Mol, None]]]:

    from_function = _FROM_FUNCTIONS[input_representation.casefold()]
    return from_function(strings)


def _mol2x(
    mols: Union[Iterable[Mol], Mol],
    output_representation: str = 'smiles',
    **kwargs
) -> Union[str, None, Iterable[Union[str, None]]]:

    to_function = _TO_FUNCTIONS[output_representation.casefold()]

    return to_function(mols, **kwargs)


def convert_string_representation(
    strings: Union[Iterable[str], str],
    input_representation: str = 'smiles', 
    output_representation: Union[Iterable[str], str] = 'smiles', 
    **kwargs
) -> Union[str, None, Iterable[Union[str, None]], Dict[str, Union[str, None, Iterable[Union[str, None]]]]]:
    
    """Convert between string representations of chemical structures.
    
    """

    mols = _x2mol(cast(strings, to=list), input_representation)
    # print_err(mols)

    if not isinstance(output_representation, str) and isinstance(output_representation, Iterable):
        mols = cast(mols, to=list)
        outstrings = {rep_name: _mol2x(mols, rep_name, **kwargs) 
                      for rep_name in output_representation}
    elif isinstance(output_representation, str):
        outstrings = _mol2x(mols, output_representation, **kwargs) 
    else:
        raise TypeError(f"Specified output representation must be a string or iterable")
    # print_err(outstrings)

    return outstrings


def _convert_input_to_smiles(f: Callable) -> Callable:

    @wraps(f)
    def _f(
        strings: Union[Iterable[str], str], 
        input_representation: str = 'smiles',
        *args, **kwargs
    ) -> Union[str, None, Iterable[Union[str, None]]]:
        
        smiles = convert_string_representation(
            cast(strings, to=list), 
            output_representation='smiles', 
            input_representation=input_representation
        )
        return f(strings=smiles, *args, **kwargs)

    return _f
