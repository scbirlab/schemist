"""Tools for processing tabular data."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from functools import partial

try:
    from itertools import batched
except ImportError:
    from carabiner.itertools import batched

from carabiner.cast import cast
from pandas import DataFrame, Index, concat

from .cleaning import clean_smiles, clean_selfies
from .converting import convert_string_representation
from .features import calculate_feature
from .generating import sample_peptides_in_length_range, react
from .splitting import split
from .typing import DataSplits

def _get_column_values(df: DataFrame, 
                       column: Union[str, List[str]]):

    try:
        column_values = df[column]
    except KeyError:
        raise KeyError(f"Column {column} does not appear to be in the data: {', '.join(df.columns)}")
    else:
        return column_values


def _get_error_tally(df: DataFrame, 
                     cols: Union[str, List[str]]) -> Dict[str, int]:

    cols = cast(cols, to=list)

    try:
        tally = {col: (df[col].isna() | ~df[col]).sum() for col in cols}
    except TypeError:
        tally = {col: df[col].isna().sum() for col in cols}

    return tally


def converter(df: DataFrame, 
              column: str = 'smiles',
              input_representation: str = 'smiles',
              output_representation: Union[str, Iterable[str]] = 'smiles',
              prefix: Optional[str] = None,
              options: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, int], DataFrame]:
    
    """
    
    """

    prefix = prefix or ''  
    options = options or {}

    column_values = _get_column_values(df, column)

    output_representation = cast(output_representation, to=list)
    converters = convert_string_representation(
        column_values,
        output_representation=output_representation,
        input_representation=input_representation,
        **options,
    )
    converted = {f"{prefix}{conversion_name}": cast(conversion, to=list) 
                 for conversion_name, conversion in converters.items()}
    df = df.assign(**converted)

    return  _get_error_tally(df, list(converted)), df


def cleaner(df: DataFrame, 
            column: str = 'smiles',
            input_representation: str = 'smiles',
            prefix: Optional[str] = None) -> Tuple[Dict[str, int], DataFrame]:
    
    """
    
    """
     
    if input_representation.casefold() == 'smiles':
        cleaner = clean_smiles
    elif input_representation.casefold() == 'selfies':
        cleaner = clean_selfies
    else:
        raise ValueError(f"Representation {input_representation} is not supported for cleaning.")
    
    prefix = prefix or ''
    new_column = f"{prefix}{column}"

    df = df.assign(**{new_column: lambda x: cast(cleaner(_get_column_values(x, column)), to=list)})

    return _get_error_tally(df, new_column), df


def featurizer(
    df: DataFrame, 
    feature_type: str,
    column: str = 'smiles',
    ids: Optional[Union[str, Iterable[str]]] = None,
    input_representation: str = 'smiles',
    prefix: Optional[str] = None
) -> Tuple[Dict[str, int], DataFrame]:
    
    """Generate a feature table based on a column of the input dataframe.

    Examples
    ========
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1,2,3], "b": ["A", "B", "C"], "smiles": ["C", "CCC", "CCCO"]})
    >>> valid, fps = featurizer(df, "fp")
    >>> fps  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            a  b ... meta_feature_type  meta_feature_valid
    smiles       ...                                      
    C       1  A ...            morgan                True
    CCC     2  B ...            morgan                True
    CCCO    3  C ...            morgan                True
    <BLANKLINE>
    [3 rows x 6 columns]
    >>> featurizer(df, "fp", ids="b")[-1]  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            b  ...  meta_feature_valid
    smiles     ...                    
    C       A  ...                True
    CCC     B  ...                True
    CCCO    C  ...                True
    <BLANKLINE>
    [3 rows x 4 columns]
    >>> featurizer(df, "fp", ids=["a", "b"])[-1]  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            a  ...  meta_feature_valid
    smiles     ...                    
    C       1  ...                True
    CCC     2  ...                True
    CCCO    3  ...                True
    <BLANKLINE>
    [3 rows x 5 columns]
    >>> featurizer(df, "2d", ids=["a", "b"])[-1]  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            a  b  ...  meta_feature_valid
    smiles        ...                    
    C       1  A  ...                True
    CCC     2  B  ...                True
    CCCO    3  C  ...                True
    <BLANKLINE>
    [3 rows x 204 columns]

    """

    if ids is None:
        ids = df.columns.tolist()
    else:
        ids = cast(ids, to=list)

    strings = _get_column_values(df, column)
    feature_df = calculate_feature(
        feature_type=feature_type,
        strings=strings, 
        prefix=prefix,
        input_representation=input_representation,
        return_dataframe=True,
    )
    feature_df = feature_df.reset_index(drop=True)
    
    if len(ids) > 0:
        feature_df = concat(
            [df[ids], feature_df], 
            axis=1,
        )
    feature_df.index = Index(strings, name=column)

    return _get_error_tally(feature_df, 'meta_feature_valid'), feature_df


def assign_groups(df: DataFrame, 
                  grouper: Callable[[Union[str, Iterable[str]]], Dict[str, Tuple[int]]],
                  group_name: str = 'group',
                  column: str = 'smiles', 
                  input_representation: str = 'smiles',
                  *args, **kwargs) -> Tuple[Dict[str, Tuple[int]], DataFrame]:
    
    group_idx = grouper(strings=_get_column_values(df, column), 
                        input_representation=input_representation,
                        *args, **kwargs)
    
    inv_group_idx = {i: group for group, idx in group_idx.items() for i in idx}
    groups = [inv_group_idx[i] for i in range(len(inv_group_idx))]

    return group_idx, df.assign(**{group_name: groups})


def _assign_splits(df: DataFrame, 
                   split_idx: DataSplits,
                   use_df_index: bool = False) -> DataFrame:

    row_index = df.index if use_df_index else tuple(range(df.shape[0]))

    df = df.assign(**{f'is_{key}': [i in getattr(split_idx, key) for i in row_index] 
                      for key in split_idx._fields})
    split_counts = {key: sum(df[f'is_{key}'].values) for key in split_idx._fields}

    return split_counts, df


def splitter(df: DataFrame, 
             split_type: str = 'random', 
             column: str = 'smiles', 
             input_representation: str = 'smiles',
             *args, **kwargs) -> Tuple[Dict[str, int], DataFrame]:
    
    """
    
    """
    
    split_idx = split(split_type=split_type,
                      strings=_get_column_values(df, column), 
                      input_representation=input_representation,
                      *args, **kwargs)
    
    return _assign_splits(df, split_idx=split_idx)


def reactor(df: DataFrame, 
            column: str = 'smiles', 
            reaction: Union[str, Iterable[str]] = 'N_to_C_cyclization', 
            prefix: Optional[str] = None,
            *args, **kwargs) -> Tuple[Dict[str, int], DataFrame]:
    
    """
    
    """

    prefix = prefix or ''
    
    reactors = {col: partial(react, reaction=col)
                for col in cast(reaction, to=list)}

    column_values = _get_column_values(df, column)

    new_columns = {f"{prefix}{col}": list(_reactor(strings=column_values, *args, **kwargs))
                   for col, _reactor in reactors.items()}
    
    df = df.assign(**new_columns)
    
    return _get_error_tally(df, reaction), df


def _peptide_table(max_length: int,
                   min_length: Optional[int] = None,
                   by: int = 1,
                   n: Optional[Union[float, int]] = None,
                   prefix: str = '',
                   suffix: str = '',
                   generator: bool = False, 
                   batch_size: int = 1000,
                   *args, **kwargs) -> Union[DataFrame, Iterable]:
    
    min_length = min_length or max_length

    peptides = sample_peptides_in_length_range(max_length=max_length,
                                               min_length=min_length,
                                               by=by,
                                               n=n,
                                               *args, **kwargs)
    
    if generator:

        return (DataFrame(dict(peptide_sequence=[f"{prefix}{pep}{suffix}" for pep in peps]))
                for peps in batched(peptides, batch_size))

    else:

        peps = [f"{prefix}{pep}{suffix}" 
                for pep in peptides]

        return DataFrame(dict(peptide_sequence=peps))