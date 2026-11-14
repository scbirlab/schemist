"""Tools for processing tabular data."""

from typing import TYPE_CHECKING, Any, Optional, Union
from collections.abc import Callable, Iterable, Mapping
from functools import partial

try:
    from itertools import batched
except ImportError:
    from carabiner.itertools import batched
from carabiner.cast import cast


if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

from .cleaning import clean_smiles, clean_selfies
from .converting import convert_string_representation
from .features import calculate_feature
from .generating import sample_peptides_in_length_range, react
from .splitting import split
from .typing import DataSplits


def _get_column_values(
    df: DataFrame, 
    column: Union[str, list[str]]
):

    try:
        column_values = df[column]
    except KeyError:
        raise KeyError(f"Column {column} does not appear to be in the data: {', '.join(df.columns)}")
    else:
        return column_values


def _get_error_tally(df: DataFrame, 
                     cols: Union[str, list[str]]) -> dict[str, int]:

    cols = cast(cols, to=list)

    try:
        tally = {col: (df[col].isna() | ~df[col]).sum() for col in cols}
    except TypeError:
        tally = {col: df[col].isna().sum() for col in cols}

    return tally


def converter(
    df: DataFrame,
    column: str = 'smiles',
    input_representation: str = 'smiles',
    output_representation: Union[str, Iterable[str]] = 'smiles',
    prefix: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    skip_desalt: bool = False,
    return_all_fragments: bool = False,
    keep_H: bool = False,
    protonate: bool = False,
    protonation_pH: float = 6.7,
) -> tuple[dict[str, int], DataFrame]:
    
    """Convert a column of chemical structures to other representations.

    Structures are cleaned before conversion; see
    :func:`schemist.converting.convert_string_representation`. With
    ``return_all_fragments``, removed fragments go in ``<prefix>smiles_salt``.

    Examples
    ========
    >>> import pandas as pd
    >>> df = pd.DataFrame({"smiles": ["CN.Cl", "CCO"]})
    >>> errors, out = converter(df, output_representation=["smiles", "inchikey"], prefix="c_", return_all_fragments=True)
    >>> out[["c_smiles", "c_smiles_salt"]].fillna("").values.tolist()
    [['CN', 'Cl'], ['CCO', '']]
    >>> list(errors)
    ['c_smiles', 'c_inchikey']

    """

    prefix = prefix or ''  
    options = options or {}

    column_values = _get_column_values(df, column)

    output_representation = cast(output_representation, to=list)
    converters = convert_string_representation(
        column_values,
        output_representation=output_representation,
        input_representation=input_representation,
        skip_desalt=skip_desalt,
        return_all_fragments=return_all_fragments,
        keep_H=keep_H,
        protonate=protonate,
        protonation_pH=protonation_pH,
        **options,
    )
    converted = {f"{prefix}{conversion_name}": cast(conversion, to=list) 
                 for conversion_name, conversion in converters.items()}
    df = df.assign(**converted)
    tallied = [f"{prefix}{name}" for name in output_representation]

    return  _get_error_tally(df, tallied), df


def cleaner(
    df: DataFrame,
    column: str = 'smiles',
    input_representation: str = 'smiles',
    prefix: Optional[str] = None,
    skip_desalt: bool = False,
    return_all_fragments: bool = False,
    keep_H: bool = False,
    protonate: bool = False,
    protonation_pH: float = 6.7,
) -> tuple[dict[str, int], DataFrame]:
    
    """Clean a column of chemical structures.

    SMILES are sanitized, desalted (unless ``skip_desalt``), stripped of explicit
    hydrogens (unless ``keep_H``) and optionally protonated at ``protonation_pH``;
    see :func:`schemist.cleaning.clean_smiles`. With ``return_all_fragments``, the
    removed fragments go in an extra column named ``<prefix><column>_salt``.

    Examples
    ========
    >>> import pandas as pd
    >>> df = pd.DataFrame({"smiles": ["CN.Cl", "CCO", "not_a_smiles"]})
    >>> errors, out = cleaner(df, prefix="clean_", return_all_fragments=True)
    >>> out[["clean_smiles", "clean_smiles_salt"]].fillna("").values.tolist()
    [['CN', 'Cl'], ['CCO', ''], ['', '']]
    >>> int(errors["clean_smiles"])
    1

    """
    prefix = prefix or ''
    new_column = f"{prefix}{column}"
    values = _get_column_values(df, column)

    if input_representation.casefold() == 'smiles':
        cleaned = cast(
            clean_smiles(
                values,
                skip_desalt=skip_desalt,
                return_all_fragments=return_all_fragments,
                keep_H=keep_H,
                protonate=protonate,
                protonation_pH=protonation_pH,
            ),
            to=list,
        )
    elif input_representation.casefold() == 'selfies':
        if skip_desalt or return_all_fragments or keep_H or protonate:
            raise ValueError("Desalting, hydrogen and protonation options are only supported for SMILES.")
        cleaned = cast(clean_selfies(values), to=list)
    else:
        raise ValueError(f"Representation {input_representation} is not supported for cleaning.")

    if return_all_fragments:
        df = df.assign(**{
            new_column: [None if g is None else g.parent for g in cleaned],
            f"{new_column}_salt": [None if g is None else g.other_smiles for g in cleaned],
        })
    else:
        df = df.assign(**{new_column: cleaned})

    return _get_error_tally(df, new_column), df


def featurizer(
    df: DataFrame, 
    feature_type: str,
    column: str = 'smiles',
    ids: str | Iterable[str] | None = None,
    input_representation: str = 'smiles',
    prefix: str | None = None,
    **kwargs
) -> tuple[dict[str, int], DataFrame]:
    
    """Generate a feature table based on a column of the input dataframe.

    Examples
    ========
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1,2,3], "b": ["A", "B", "C"], "smiles": ["C", "CCC", "CCCO"]})
    >>> valid, fps = featurizer(df, "fp")
    >>> fps.shape
    (3, 5)
    >>> featurizer(df, "fp", ids="b")[-1].shape
    (3, 3)
    >>> featurizer(df, "fp", ids=["a", "b"])[-1].shape
    (3, 4)
    >>> featurizer(df, "2d", ids=["a", "b"])[-1].shape
    (3, 203)

    """
    from pandas import Index, concat

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
        **kwargs,
    )
    feature_df = feature_df.reset_index(drop=True)
    
    if len(ids) > 0:
        feature_df = concat(
            [df[ids].reset_index(drop=True), feature_df], 
            axis=1,
        )
    feature_df.index = Index(strings, name=column)

    return _get_error_tally(feature_df, 'meta_feature_valid_' + feature_type), feature_df


def assign_groups(df: DataFrame, 
                  grouper: Callable[[Union[str, Iterable[str]]], dict[str, tuple[int]]],
                  group_name: str = 'group',
                  column: str = 'smiles', 
                  input_representation: str = 'smiles',
                  *args, **kwargs) -> tuple[dict[str, tuple[int]], DataFrame]:
    
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
             *args, **kwargs) -> tuple[dict[str, int], DataFrame]:
    
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
            *args, **kwargs) -> tuple[dict[str, int], DataFrame]:
    
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