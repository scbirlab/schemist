"""Tools for generating chemical features."""

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from functools import cache, wraps

from carabiner.cast import cast
from carabiner.decorators import return_none_on_error, vectorize
from descriptastorus.descriptors import MakeGenerator
from pandas import DataFrame, Series
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Mol

try:
    from rdkit.Chem.AllChem import FingeprintGenerator64 as FingerprintGenerator64, GetMorganGenerator
except ImportError: # typo in some rdkit versions
    from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator

from .cleaning import clean_smiles
from .converting import _x2mol, _mol2x, _smiles2mol, _convert_input_to_smiles

def _feature_matrix(f: Callable[[Any], DataFrame]) -> Callable[[Any], Union[DataFrame, Tuple[np.ndarray, np.ndarray]]]:

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
    generator: str = "RDKit2DHistogramNormalized"
) -> Union[DataFrame, Tuple[np.ndarray, List[str]]]:

    generator = MakeGenerator((generator, ))
    smiles = cast(clean_smiles(smiles), to=list)
    mols = cast(_smiles2mol(smiles), to=list)
    features = generator.processMols(
        mols, 
        smiles,
    )
    return np.stack(features, axis=0), [col for col, _ in generator.GetColumns()]


@_feature_matrix
@_convert_input_to_smiles
def calculate_2d_features(
    strings: Union[Iterable[str], str], 
    normalized: bool = True, 
    histogram_normalized: bool = True,
    return_dataframe: bool = False,
    *args, **kwargs
) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:

    """Calculate 2d features from string representation.

    Parameters
    ----------
    strings : str
        Input string representation(s).
    input_representation : str
        Representation type
    normalized : bool, optional
        Whether to return normalized features. Default: `True`.
    histogram_normalized : bool, optional
        Whether to return histogram normalized features (faster). Default: `True`.
    return_dataframe : bool, optional
        Whether to retrun a Pandas DataFrame instead of a numpy Array. Default: `False`.

    Returns
    -------
    DataFrame, Tuple of numpy Arrays
        If `return_dataframe = True`, a DataFrame with named feature columns, and 
        the final column called `"meta_feature_valid"` being the validity indicator.
        Otherwise returns a tuple of Arrays with the first being the matrix of 
        features and the second being the vector of validity indicators.

    Examples
    --------
    >>> features, validity = calculate_2d_features(strings='CCC')
    >>> features[:,:3]
    array([[4.22879602e-01, 1.30009101e-04, 2.00014001e-05]])
    >>> validity
    array([1.])
    >>> features, validity = calculate_2d_features(strings=['CCC', 'CCCO'])
    >>> features[:,:3]
    array([[4.22879602e-01, 1.30009101e-04, 2.00014001e-05],
           [7.38891722e-01, 6.00042003e-04, 5.00035002e-05]])
    >>> validity
    array([1., 1.])
    >>> calculate_2d_features(strings=['CCC', 'CCCO'], return_dataframe=True).meta_feature_valid
    CCC     True
    CCCO    True
    Name: meta_feature_valid, dtype: bool
    >>> ## Unusal valence
    >>> s = "O=S(=O)(OCC1OC(OC2(COS(=O)(=O)O[AlH3](O)O)OC(COS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C2OS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C1OS(=O)(=O)O[AlH3](O)O)O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O"
    >>> calculate_2d_features(strings=s)[0].shape
    (1, 200)
    >>> s = 'CCc1c(C(=O)N2CC(c3nnc4c3CCC4)C2)nc(C)c1C(=O)OC'
    >>> calculate_2d_features(strings=s)[1]
    array([1.])

    """  

    if normalized:
        if histogram_normalized:
            generator_name = "RDKit2DHistogramNormalized"
        else:
            generator_name = "RDKit2DNormalized"
    else:
        generator_name = "RDKit2D"
    
    strings = cast(strings, to=list)
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


@cache
def smiles_to_3d(
    smiles: str,
    seed: int = 42
) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Mol
    from rdkit.Chem.rdchem import AtomValenceException
    from rdkit.Chem.Descriptors3D import CalcMolDescriptors3D
    mol = _smiles2mol(clean_smiles(smiles))
    if mol is not None:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, params)
            desc = CalcMolDescriptors3D(mol)
        except (ValueError, AtomValenceException):
            desc = {"meta_feature_valid": False}
        else:
            desc["meta_feature_valid"] = True
        return desc
    else:
        return {"meta_feature_valid": False}


@_feature_matrix
@_convert_input_to_smiles
def calculate_3d_features(
    strings: Union[Iterable[str], str],
    seed: int = 42,
    return_dataframe: bool = False,
    *args, **kwargs
) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:

    """Calculate 3d features from string representation.

    Parameters
    ----------
    strings : str
        Input string representation(s).
    input_representation : str
        Representation type
    seed : int
        Seed for reproducible randomness
    return_dataframe : bool, optional
        Whether to retrun a Pandas DataFrame instead of a numpy Array. Default: `False`.

    Returns
    -------
    DataFrame, Tuple of numpy Arrays
        If `return_dataframe = True`, a DataFrame with named feature columns, and 
        the final column called `"meta_feature_valid"` being the validity indicator.
        Otherwise returns a tuple of Arrays with the first being the matrix of 
        features and the second being the vector of validity indicators.

    Examples
    --------
    >>> features, validity = calculate_3d_features(strings='CCC')
    >>> features.shape
    (1, 11)
    >>> sum(validity)
    1
    >>> features, validity = calculate_3d_features(strings=['CCC', 'CCCO'])
    >>> features.shape
    (2, 11)
    >>> sum(validity)
    2
    >>> calculate_3d_features(strings=['CCC', 'CCCO'], return_dataframe=True).meta_feature_valid
    CCC     True
    CCCO    True
    Name: meta_feature_valid, dtype: bool
    >>> ## Unusal valence
    >>> s = "O=S(=O)(OCC1OC(OC2(COS(=O)(=O)O[AlH3](O)O)OC(COS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C2OS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C(OS(=O)(=O)O[AlH3](O)O)C1OS(=O)(=O)O[AlH3](O)O)O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O.O[AlH3](O)O"
    >>> calculate_3d_features(strings=s)[0].shape
    (1, 0)
    >>> s = 'CCc1c(C(=O)N2CC(c3nnc4c3CCC4)C2)nc(C)c1C(=O)OC'
    >>> sum(calculate_3d_features(strings=s)[1])
    1

    """ 
    
    import pandas as pd
    
    strings = cast(strings, to=list)
    descriptors = [
        smiles_to_3d(s) for s in strings
    ]
            
    feature_matrix = DataFrame(descriptors).fillna(0.)
    if return_dataframe:
        feature_matrix.index = strings
        feature_matrix = (
            feature_matrix
            .assign(meta_feature_type="RDKit3D")
        )
        return feature_matrix
    else:
        feature_matrix = pd.concat([
            feature_matrix[["meta_feature_valid"]],
            feature_matrix[[col for col in feature_matrix if col != "meta_feature_valid"]],
        ], axis=1).values
        return feature_matrix[:,1:], feature_matrix[:,0]


def _fast_fingerprint(
    generator: FingerprintGenerator64, 
    mol: Mol,
    to_np: bool = True
) -> Union[str, np.ndarray]:

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
    return_dataframe: bool = False,
    *args, **kwargs
) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Calculate the binary fingerprint of string representation(s).

    Only Morgan fingerprints are allowed.

    Parameters
    ----------
    strings : str
        Input string representation(s).
    input_representation : str
        Representation type
    fp_type : str, opional
        Which fingerprint type to calculate. Default: `'morgan'`.
    radius : int, optional
        Atom radius for fingerprints. Default: `2`.
    chiral : bool, optional
        Whether to take chirality into account. Default: `True`.
    on_bits : bool, optional
        Whether to return the non-zero indices instead of the full binary vector. Default: `True`.
    return_dataframe : bool, optional
        Whether to retrun a Pandas DataFrame instead of a numpy Array. Default: `False`.

    Returns
    -------
    DataFrame, Tuple of numpy Arrays
        If `return_dataframe = True`, a DataFrame with named feature columns, and 
        the final column called `"meta_feature_valid"` being the validity indicator.
        Otherwise returns a tuple of Arrays with the first being the matrix of 
        features and the second being the vector of validity indicators.

    Raises
    ------
    NotImplementedError
        If `fp_type` is not `'morgan'`.
    
    Examples
    --------
    >>> bits, validity = calculate_fingerprints(strings='CCC')
    >>> bits.tolist()
    [['80;294;1057;1344']]
    >>> sum(validity)  # doctest: +NORMALIZE_WHITESPACE
    1 
    >>> bits, validity = calculate_fingerprints(strings=['CCC', 'CCCO'])
    >>> bits.tolist()
    [['80;294;1057;1344'], ['80;222;294;473;794;807;1057;1277']]
    >>> sum(validity)  # doctest: +NORMALIZE_WHITESPACE
    2
    >>> np.sum(calculate_fingerprints(strings=['CCC', 'CCCO'], on_bits=False)[0], axis=-1)
    array([4, 8])
    >>> calculate_fingerprints(strings=['CCC', 'CCCO'], return_dataframe=True).meta_feature_valid
    CCC     True
    CCCO    True
    Name: meta_feature_valid, dtype: bool

    """
    
    if fp_type.casefold() == 'morgan':
        generator_class = GetMorganGenerator
    else:
        raise NotImplementedError(f"Fingerprint type {fp_type} not supported!")
    
    fp_generator = generator_class(
        radius=radius, 
        includeChirality=chiral,
    )
    try:
        fp_size = fp_generator.GetOptions().fpSize
    except AttributeError:  # 'FingerprintGenerator64' object has no attribute 'GetOptions' in older rdkit versions (e.g.2022.9.5)
        test_smiles = "CCCC"
        fp_size = _fast_fingerprint(
            fp_generator, 
            _smiles2mol(test_smiles), 
            to_np=True,
        ).size
    strings = cast(strings, to=list)
    mols = (_smiles2mol(s) for s in strings)
    fp_strings = (_fast_fingerprint(fp_generator, mol, to_np=on_bits) 
                  for mol in mols)

    if on_bits:

        fingerprints = (
            map(str, np.flatnonzero(fp_string).tolist()) 
            for fp_string in fp_strings
        )
        fingerprints = [[';'.join(fp)] for fp in fingerprints]
        validity = [len(fp[0]) > 0 for fp in fingerprints]
    
    else:
        
        fingerprints = [
            np.array([
                int(digit) for digit in fp_string
            ]) 
            if fp_string is not None 
            else np.zeros((fp_size, ))
            for fp_string in fp_strings
        ]
        validity = [np.all(fp >= 0) for fp in fingerprints]
        
    feature_matrix = np.stack(fingerprints, axis=0)

    if return_dataframe:
        if feature_matrix.ndim == 1:  # on_bits only
            feature_matrix = DataFrame(
                feature_matrix, 
                columns=['fp_bits'],
                index=strings,
            )
        else:
            feature_matrix = DataFrame(
                feature_matrix,
                columns=[f"fp_{i}" for i, _ in enumerate(feature_matrix.T)],
                index=strings,
            )
        return feature_matrix.assign(
            meta_feature_type=fp_type.casefold(), 
            meta_feature_valid=validity,
        )
    else:
        return feature_matrix, validity


_FEATURE_CALCULATORS = {
    "2d": calculate_2d_features, 
    "3d": calculate_3d_features, 
    "fp": calculate_fingerprints,
}

def calculate_feature(
    feature_type: Union[str, Iterable[str]] = "all",
    return_dataframe: bool = False,
    *args, **kwargs) -> Union[DataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Calculate the binary fingerprint or descriptor vector of string representation(s).
    
    Examples
    ========
    >>> calculate_feature("2d", strings=['CCC', 'CCO'])[0].shape
    (2, 200)
    >>> calculate_feature("3d", strings=['CCC', 'CCO'])[0].shape
    (2, 11)
    >>> calculate_feature("fp", on_bits=False, strings=['CCC', 'CCO'])[0].shape
    (2, 2048)
    >>> calculate_feature("all", on_bits=False, strings=['CCC', 'CCO'])[0].shape
    (2, 2259)

    """
    if feature_type == "all":
        feature_type = list(_FEATURE_CALCULATORS)
    feature_type = cast(feature_type, to=list)
    featurizers = {_type: _FEATURE_CALCULATORS[_type] for _type in feature_type}
    dfs = {
        _type: f(*args, return_dataframe=return_dataframe, **kwargs)
        for _type, f in featurizers.items()
    }
    if return_dataframe:
        import pandas as pd
        return pd.concat([
            _df.drop(columns=["meta_feature_valid", "meta_feature_type"])
            for _df in dfs.values()
        ] + [
            _df[["meta_feature_valid"]].rename(columns={"meta_feature_valid": f"meta_feature_valid_{_type}"})
            for _type, _df in dfs.items()
        ], axis=1)
    else:
        return np.concatenate([_df[0] for _df in dfs.values()], axis=1), np.stack([_df[1] for _df in dfs.values()], axis=0).astype(int)
