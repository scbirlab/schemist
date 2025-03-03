"""Chemical structure cleaning routines."""

from carabiner.decorators import return_none_on_error, vectorize
from rdkit.Chem import (
    Mol,
    MolFromSmiles,
    MolToSmiles,
)
import selfies as sf

from .sanifix5 import sanifix

# @return_none_on_error
def sanitize_smiles_to_mol(s: str) -> Mol:
    """Apply sanifix5.
    
    """
    m = MolFromSmiles(s, sanitize=False)
    return sanifix(m)


@vectorize
def clean_smiles(smiles: str, 
                 *args, **kwargs) -> str:
    """Sanitize a SMILES string or list of SMILES strings.
    
    """
    return MolToSmiles(sanitize_smiles_to_mol(smiles), *args, **kwargs)


@vectorize
def clean_selfies(selfies: str, 
                  *args, **kwargs) -> str:
    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """
    return sf.encode(MolToSmiles(sanitize_smiles_to_mol(sf.decode(selfies), *args, **kwargs)))