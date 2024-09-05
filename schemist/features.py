"""Tools for generating chemical features."""

from typing import Any, Callable, Iterable, List, Optional, Union
from functools import wraps

from descriptastorus.descriptors import MakeGenerator
from pandas import DataFrame, Series
import numpy as np
from rdkit.Chem.AllChem import FingeprintGenerator64, GetMorganGenerator, Mol

from .converting import _smiles2mol, _convert_input_to_smiles

def _feature_matrix(f: Callable[[Any], DataFrame]) -> Callable[[Any], DataFrame]:

    @wraps(f)
    def _f(prefix: Optional[str] = None,
           *args, **kwargs) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:

        feature_matrix = f(*args, **kwargs)

        if prefix is not None and isinstance(feature_matrix, DataFrame):
            new_cols = {col: f"{prefix}_{col}" 
                        for col in feature_matrix.columns 
                        if not col.startswith('_meta')}
            feature_matrix = feature_matrix.rename(columns=new_cols)

        return feature_matrix

    return _f


def _get_descriptastorus_features(
    smiles: Iterable[str], 
    generator: str
) -> Union[DataFrame, Tuple[np.ndarray, List[str]]]:

    generator = MakeGenerator((generator, ))
    features = map(generator.process, smiles)    
    return np.stack(features, axis=0), [col for col, _ in generator.GetColumns()]


@_feature_matrix
@_convert_input_to_smiles
def calculate_2d_features(
    strings: Union[Iterable[str], str], 
    normalized: bool = True, 
    histogram_normalized: bool = True,
    return_dataframe: bool = False
) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:

    """Calculate 2d features from string representation.
    
    """  

    if normalized:
        if histogram_normalized:
            generator_name = "RDKit2DHistogramNormalized"
        else:
            generator_name = "RDKit2DNormalized"
    else:
        generator_name = "RDKit2D"

    feature_matrix, columns = _get_descriptastorus_features(
        strings,
        generator=generator_name,
    )

    if return_dataframe:
        feature_matrix = DataFrame(
            feature_matrix, 
            index=strings,
            columns=columns,
        )

        feature_matrix = (
            feature_matrix
            .rename(columns={f"{generator_name}_calculated": "meta_feature_valid0"})
            .assign(meta_feature_type=generator_name, 
                    meta_feature_valid=lambda x: (x['meta_feature_valid0'] == 1.))
            .drop(columns=['meta_feature_valid0'])
        )
        return feature_matrix
    else:
        return feature_matrix[:,1:], feature_matrix[:,0]


def _fast_fingerprint(generator: FingeprintGenerator64, 
                      mol: Mol,
                      to_np: bool = True) -> Union[str, np.ndarray]:

    try:
        fp_string = generator.GetFingerprint(mol).ToBitString()
    except:
        return None
    else:
        if to_np:
            return np.frombuffer(fp_string.encode(), 'u1') - ord('0')
        else:
            return fp_string
    

@_feature_matrix
@_convert_input_to_smiles
def calculate_fingerprints(
    strings: Union[Iterable[str], str], 
    fp_type: str = 'morgan',
    radius: int = 2,
    chiral: bool = True, 
    on_bits: bool = True,
    return_dataframe: bool = False
) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Calculate the binary fingerprint of string representation(s).
    
    """
    
    if fp_type.casefold() == 'morgan':
        generator_class = GetMorganGenerator
    else:
        raise NotImplementedError(f"Fingerprint type {fp_type} not supported!")
    
    fp_generator = generator_class(radius=radius, 
                                   includeChirality=chiral)
    mols = (_smiles2mol(s) for s in strings)
    fp_strings = (_fast_fingerprint(fp_generator, mol, to_np=on_bits) 
                  for mol in mols)

    if on_bits:

        fingerprints = (map(str, np.flatnonzero(fp_string).tolist()) 
                        for fp_string in fp_strings)
        fingerprints = [';'.join(fp) for fp in fingerprints]
        validity = [len(fp) > 0 for fp in fingerprints]
        feature_matrix = fingerprints
    
    else:
        
        fingerprints = [np.array([int(digit) for digit in fp_string]) 
                        if fp_string is not None 
                        else (-np.ones((fp_generator.GetOptions().fpSize, )))
                        for fp_string in fp_strings]
        validity = [np.all(fp >= 0) for fp in fingerprints]
        feature_matrix = np.stack(fingerprints, axis=0)

    if return_dataframe:
        ncol = feature_matrix.shape[-1]
        if ncol == 1:
            feature_matrix = DataFrame(feature_matrix, 
                                       columns=['fp_bits'])
        else:
            feature_matrix = DataFrame(feature_matrix,
                                        columns=[f"fp_{i}" for i in range(ncol)])
        return feature_matrix.assign(meta_feature_type=fp_type.casefold(), 
                                     meta_feature_valid=validity)
    else:
        return feature_matrix, validity


_FEATURE_CALCULATORS = {
    "2d": calculate_2d_features, 
    "fp": calculate_fingerprints,
}

def calculate_feature(feature_type: str,
                      *args, **kwargs) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Calculate the binary fingerprint or descriptor vector of string representation(s).
    
    """
    
    featurizer = _FEATURE_CALCULATORS[feature_type]
    return featurizer(*args, **kwargs)