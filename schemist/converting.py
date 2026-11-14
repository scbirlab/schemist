"""Converting between chemical representation formats."""

from typing import TYPE_CHECKING, Any, Optional, Union
from collections.abc import Callable, Iterable
from functools import wraps

from carabiner import print_err
from carabiner.cast import cast, flatten
from carabiner.decorators import return_none_on_error, vectorize
from carabiner.itertools import batched

# from datamol import sanitize_smiles
import nemony as nm

from rdkit.Chem import (
    Crippen, 
    Descriptors, 
    rdMolDescriptors,
    Mol, 
    MolFromInchi, 
    MolFromHELM, 
    MolFromSequence,
    MolToInchi, 
    MolToInchiKey, 
    MolToSmiles,
)
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

import selfies as sf

if TYPE_CHECKING:
    from requests import Session
else:
    Session = Any

from .cleaning import clean_mol, sanitize_smiles_to_mol
from .rest_lookup import _inchikey2pubchem_name_id, _inchikey2cactus_name

@vectorize
@return_none_on_error
def _seq2mol(s: str) -> OptionalMol:

    return MolFromSequence(s, sanitize=True)


@vectorize
@return_none_on_error
def _helm2mol(s: str) -> OptionalMol:

    return MolFromHELM(s, sanitize=True)


def mini_helm2helm(s: str) -> list[str]:

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


def _mol2pubchem(
    m: MolOrIterable,
    session: Session | None = None,
    chunksize: int = 16
) -> list[dict[str, None | int | str]]:
    
    inchikeys = cast(_mol2inchikey(m), to=list)
    pubchem_ids = []

    for _inchikeys in batched(inchikeys, chunksize):
        pubchem_ids += _inchikey2pubchem_name_id(
            _inchikeys,
        )

    return pubchem_ids


@return_none_on_error
def _mol2pubchem_id(
    m: MolOrIterable,
    session: Session | None = None,
    chunksize: int = 32,
    **kwargs
) -> str | list[str]:

    return flatten([val['pubchem_id'] 
                     for val in _mol2pubchem(m, 
                                             session=session, 
                                             chunksize=chunksize)])


@return_none_on_error
def _mol2pubchem_name(
    m: MolOrIterable,
    session: Session | None = None,
    chunksize: int = 32,
    **kwargs
) -> str | list[str]:
    return flatten([
        val['pubchem_name'] 
        for val in _mol2pubchem(
            m, 
            session=session, 
            chunksize=chunksize,
        )
    ])

@return_none_on_error
def _mol2cactus_name(
    m: MolOrIterable,
    session: Session | None = None,
    **kwargs
) -> str | list[str]:
    return _inchikey2cactus_name(
        _mol2inchikey(m), 
        session=session,
    )


@vectorize
@return_none_on_error
def _mol2scaffold(
    m: Mol,
    chiral: bool = True,
    **kwargs,
) -> str:
    return MurckoScaffoldSmiles(
        mol=m, 
        includeChirality=chiral,
    )


@vectorize
@return_none_on_error
def _mol2selfies(m: Mol,
                 **kwargs) -> str:
    s = sf.encoder(_mol2isomeric_canonical_smiles(m))
    return s if s != -1 else None


_TO_FUNCTIONS = {
    "smiles": _mol2isomeric_canonical_smiles,
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
    "max_charge": _mol2max_charge,
}

_FROM_FUNCTIONS = {
    "smiles": _smiles2mol,
    "selfies": _selfies2mol,
    "inchi": _inchi2mol,
    "aa_seq": _seq2mol,
    "helm": _helm2mol,
    "minihelm": _mini_helm2mol,
}


def _x2mol(
    strings: StrOrIterable,
    input_representation: str = 'smiles'
) -> OptionalMolOrIterable:
    from_function = _FROM_FUNCTIONS[input_representation.casefold()]
    return from_function(strings)


def _mol2x(
    mols: Union[Iterable[Mol], Mol],
    output_representation: str = 'smiles',
    **kwargs
) -> OptionalStrOrIterable:
    to_function = _TO_FUNCTIONS[output_representation.casefold()]
    return to_function(mols, **kwargs)


def convert_string_representation(
    strings: Iterable[str] | str,
    input_representation: str = 'smiles', 
    output_representation: Iterable[str] | str = 'smiles', 
    skip_desalt: bool = False,
    return_all_fragments: bool = False,
    keep_H: bool = False,
    protonate: bool = False,
    protonation_pH: float = 6.7,
    **kwargs
) -> OptionalStrOrIterable | dict[str, OptionalStrOrIterable]:
    
    r"""Convert between string representations of chemical structures.

    Every input representation is parsed to a ``Mol`` and cleaned with
    :func:`schemist.cleaning.clean_mol` (desalting, explicit-hydrogen removal,
    optional protonation) before conversion, so all outputs describe the same
    cleaned structure.

    Parameters
    ==========
    strings : str or Iterable[str]
        Input structures.
    input_representation : str, optional
        One of the keys of ``_FROM_FUNCTIONS``. Default: ``"smiles"``.
    output_representation : str or Iterable[str], optional
        One or more keys of ``_TO_FUNCTIONS``. Default: ``"smiles"``.
    skip_desalt, keep_H, protonate, protonation_pH
        Passed to :func:`schemist.cleaning.clean_mol`.
    return_all_fragments : bool, optional
        Also return the removed fragments as dot-separated SMILES under the key
        ``"smiles_salt"``. Forces dictionary output. Default: ``False``.
    **kwargs
        Passed to the output conversion functions.

    Returns
    =======
    Iterable or dict
        Converted strings, or a dict of them keyed by output representation.

    Examples
    ========
    >>> list(convert_string_representation(["CN.Cl", "C/C=C\\C"], output_representation="inchikey"))
    ['BAVYZALUXZFZLV-UHFFFAOYSA-N', 'IAQRGUVFOMOMEM-ARJAWSKDSA-N']
    >>> convert_string_representation("CN.Cl", skip_desalt=True)
    'CN.Cl'
    >>> convert_string_representation("CN.Cl", output_representation="smiles", return_all_fragments=True)
    {'smiles': 'CN', 'smiles_salt': 'Cl'}
    >>> out = convert_string_representation(
    ...     ["CN.Cl", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"],
    ...     input_representation="inchi",
    ...     output_representation="smiles",
    ...     return_all_fragments=True,
    ... )
    >>> {k: list(v) for k, v in out.items()}
    {'smiles': [None, 'CCO'], 'smiles_salt': [None, None]}

    """

    mols = cast(_x2mol(cast(strings, to=list), input_representation), to=list)
    groups = [
        clean_mol(
            mol,
            skip_desalt=skip_desalt,
            keep_H=keep_H,
            protonate=protonate,
            protonation_pH=protonation_pH,
        )
        for mol in mols
    ]
    mols = [None if g is None else g.parent for g in groups]

    if return_all_fragments and isinstance(output_representation, str):
        output_representation = [output_representation]

    if not isinstance(output_representation, str) and isinstance(output_representation, Iterable):
        outstrings = {rep_name: _mol2x(mols, rep_name, **kwargs) 
                      for rep_name in output_representation}
        if return_all_fragments:
            salts = [None if g is None else g.other_smiles for g in groups]
            # match vectorize: a single input gives scalar outputs
            outstrings["smiles_salt"] = salts[0] if len(salts) == 1 else salts
    elif isinstance(output_representation, str):
        outstrings = _mol2x(mols, output_representation, **kwargs) 
    else:
        raise TypeError(f"Specified output representation must be a string or iterable")

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
