"""Chemical structure cleaning routines."""

from typing import TYPE_CHECKING, Any
from carabiner.decorators import return_none_on_error, vectorize

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

# @return_none_on_error
def sanitize_smiles_to_mol(s: str) -> Mol:
    r"""Parse SMILES without strict sanitization, apply sanifix5, then perceive stereochemistry.

    Parsing with sanitize=False skips the stereochemistry perception that
    MolFromSmiles normally performs, so double-bond (E/Z) stereo was left
    unassigned and chirality-aware fingerprints silently differed from RDKit's.
    Stereochemistry is now assigned after sanitization, as MolFromSmiles does.

    Examples
    ========
    >>> from rdkit.Chem import BondStereo, MolFromSmiles
    >>> m = sanitize_smiles_to_mol("C/C=C/C")
    >>> m.GetBondWithIdx(1).GetStereo() == BondStereo.STEREOE
    True
    >>> m = sanitize_smiles_to_mol("C/C=C\C")
    >>> m.GetBondWithIdx(1).GetStereo() == BondStereo.STEREOZ
    True
    >>> m = sanitize_smiles_to_mol("C[C@H](N)C(=O)O")
    >>> m.GetAtomWithIdx(1).GetProp("_CIPCode")
    'S'
    >>> from rdkit.Chem import rdFingerprintGenerator
    >>> g = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    >>> s = "CC(=O)N/C=C/c1ccccc1"
    >>> g.GetFingerprint(sanitize_smiles_to_mol(s)) == g.GetFingerprint(MolFromSmiles(s))
    True

    """
    from rdkit.Chem import AssignStereochemistry, MolFromSmiles
    from .sanifix5 import sanifix
    m = sanifix(MolFromSmiles(s, sanitize=False))
    if m is not None:
        AssignStereochemistry(m, cleanIt=True, force=True)
    return m


@vectorize
@return_none_on_error
def clean_smiles(
    smiles: str, 
    *args, 
    **kwargs
) -> str:
    """Sanitize a SMILES string or list of SMILES strings.
    
    """
    from rdkit.Chem import MolToSmiles
    return MolToSmiles(
        sanitize_smiles_to_mol(smiles), 
        *args, 
        **kwargs)


@vectorize
def clean_selfies(
    selfies: str, 
    *args, 
    **kwargs
) -> str:
    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """
    from rdkit.Chem import MolToSmiles
    import selfies as sf
    return sf.encode(MolToSmiles(sanitize_smiles_to_mol(sf.decode(selfies), *args, **kwargs)))
