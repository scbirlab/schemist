"""Chemical structure cleaning routines."""

from typing import TYPE_CHECKING, Any
from carabiner.decorators import return_none_on_error, vectorize

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

# @return_none_on_error
def sanitize_smiles_to_mol(s: str) -> Mol:
    """Apply sanifix5.
    
    """
    from rdkit.Chem import MolFromSmiles
    from .sanifix5 import sanifix
    m = MolFromSmiles(s, sanitize=False)
    return sanifix(m)


@vectorize
@return_none_on_error
def clean_smiles(smiles: str, 
                 *args, **kwargs) -> str:
    """Sanitize a SMILES string or list of SMILES strings.
    
    """
    from rdkit.Chem import MolToSmiles
    return MolToSmiles(sanitize_smiles_to_mol(smiles), *args, **kwargs)


@vectorize
def clean_selfies(selfies: str, 
                  *args, **kwargs) -> str:
    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """
    from rdkit.Chem import MolToSmiles
    import selfies as sf
    return sf.encode(MolToSmiles(sanitize_smiles_to_mol(sf.decode(selfies), *args, **kwargs)))
