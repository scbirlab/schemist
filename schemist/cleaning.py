"""Chemical structure cleaning routines."""

from carabiner.decorators import vectorize

from datamol import sanitize_smiles
import selfies as sf

@vectorize
def clean_smiles(smiles: str, 
                 *args, **kwargs) -> str:

    """Sanitize a SMILES string or list of SMILES strings.
    
    """

    return sanitize_smiles(smiles, *args, **kwargs) 


@vectorize
def clean_selfies(selfies: str, 
                  *args, **kwargs) -> str:

    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """

    return sf.encode(sanitize_smiles(sf.decode(selfies), *args, **kwargs))