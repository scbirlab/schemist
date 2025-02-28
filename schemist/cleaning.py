"""Chemical structure cleaning routines."""

from carabiner.decorators import vectorize

from rdkit.Chem import MolToSmiles
import selfies as sf

from .converting import sanitize_smiles_to_mol

@vectorize
def clean_smiles(smiles: str, 
                 *args, **kwargs) -> str:

    """Sanitize a SMILES string or list of SMILES strings.
    
    """

    return MolToSmiles(sanitize_smiles_to_mol(smiles, *args, **kwargs))


@vectorize
def clean_selfies(selfies: str, 
                  *args, **kwargs) -> str:

    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """

    return sf.encode(MolToSmiles(sanitize_smiles_to_mol(sf.decode(selfies), *args, **kwargs)))