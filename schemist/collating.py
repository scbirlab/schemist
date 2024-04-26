"""Tools to collate chemical data files."""

from typing import Callable, Dict, Iterable, List, Optional, Tuple, TextIO, Union

from collections import Counter
from functools import partial
from glob import glob
import os

from carabiner.pd import read_table, resolve_delim
from carabiner import print_err
import numpy as np
from pandas import DataFrame, concat

from .converting import convert_string_representation, _FROM_FUNCTIONS
from .io import FILE_READERS

GROUPING_COLUMNS = ("filename", "file_format", "library_name", "string_representation")
ESSENTIAL_COLUMNS = GROUPING_COLUMNS + ("compound_collection", "plate_id", "well_id")

def _column_mapper(df: DataFrame, 
                   cols: Iterable[str]) -> Tuple[Callable, Dict]:

    basic_map = {column: df[column].tolist()[0] for column in cols}
    inv_basic_map = {value: key for key, value in basic_map.items()}

    def column_mapper(x: DataFrame) -> DataFrame:

        new_df = DataFrame()

        for new_col, old_col in basic_map.items():

            # old_col = str(old_col)

            if old_col is None or str(old_col) in ('None', 'nan', 'NA'):

                new_df[new_col] = None
            
            elif '+' in old_col:
                
                splits = old_col.split('+')
                new_df[new_col] = x[splits[0]].str.cat([x[s].astype(str) 
                                                        for s in splits[1:]])

            elif ';' in old_col:

                col, char, index = old_col.split(';')
                index = [int(i) for i in index.split(':')]

                if len(index) == 1:
                    index = slice(index[0], index[0] + 1)
                else:
                    index = slice(*index)
                
                try:

                    new_df[new_col] = (x[col]
                                       .str.split(char)
                                       .map(lambda y: char.join(y[index] if y is not np.nan else []))
                                       .str.strip())

                except TypeError as e:

                    print_err(x[col].str.split(char))

                    raise e

            else:
                try:
                    new_df[new_col] = x[old_col].copy()
                except KeyError:
                    raise KeyError(f"Column {old_col} mapped to {new_col} is not in the input data: " + ", ".join(x.columns))

        return new_df
       
    return column_mapper, inv_basic_map


def _check_catalog(catalog: DataFrame,
                   catalog_smiles_column: str = 'input_smiles') -> None:

    essential_columns = (catalog_smiles_column, ) + ESSENTIAL_COLUMNS
    missing_essential_cols = [col for col in essential_columns 
                              if col not in catalog]

    if len(missing_essential_cols) > 0:

        print_err(catalog.columns.tolist())

        raise KeyError("Missing required columns from catalog: " + 
                       ", ".join(missing_essential_cols))
    
    return None


def collate_inventory(catalog: DataFrame,
                      root_dir: Optional[str] = None,
                      drop_invalid: bool = True,
                      drop_unmapped: bool = False,
                      catalog_smiles_column: str = 'input_smiles',
                      id_column_name: Optional[str] = None,
                      id_n_digits: int = 8,
                      id_prefix: str = '') -> DataFrame:
    
    f"""Process a catalog of files containing chemical libraries into a uniform dataframe.

    The catalog table needs to have columns {', '.join(ESSENTIAL_COLUMNS)}:

    - filename is a glob pattern of files to collate
    - file_format is one of {', '.join(FILE_READERS.keys())}
    - smiles_column contains smiles strings

    Other columns are optional and can have any name, but must contain the name or a pattern
    matching a column (for tabular data) or field (for SDF data) in the files 
    of the `filename` column. In the output DataFrame, the named column data will be mapped.

    Optional column contents can be either concatenated or split using the following
    pattern:

    - col1+col2: concatenates the contents of `col1` and `col2`
    - col1;-;1:2 : splits the contents of `col1` on the `-` character, and takes splits 1-2 (0-indexed)

    Parameters
    ----------
    catalog : pd.DataFrame
        Table cataloging locations and format of data. Requires 
        columns {', '.join(ESSENTIAL_COLUMNS)}.
    root_dir : str, optional
        Path to look for data files. Default: current directory.
    drop_invalid : bool, optional
        Whether to drop rows containing invalid SMILES.
    

    Returns
    -------
    pd.DataFrame
        Collated chemical data.
    
    """

    root_dir = root_dir or '.'
    
    _check_catalog(catalog, catalog_smiles_column)
    
    nongroup_columns = [col for col in catalog 
                         if col not in GROUPING_COLUMNS]
    loaded_dataframes = []
    report = Counter({"invalid SMILES": 0, 
                      "rows processed": 0})

    grouped_catalog = catalog.groupby(list(GROUPING_COLUMNS))
    for (this_glob, this_filetype, 
         this_library_name, this_representation), filename_df in grouped_catalog:

        print_err(f'\nProcessing {this_glob}:')

        this_glob = glob(os.path.join(root_dir, this_glob))

        these_filenames = sorted(f for f in this_glob 
                                 if not os.path.basename(f).startswith('~$'))
        print_err('\t- ' + '\n\t- '.join(these_filenames))
        
        column_mapper, mapped_cols = _column_mapper(filename_df, 
                                                    nongroup_columns)

        reader = FILE_READERS.get(this_filetype, read_table)

        for filename in these_filenames:
            
            this_data0 = reader(filename)

            if not drop_unmapped:
                unmapped_cols = {col: 'x_' + col.casefold().replace(' ', '_') 
                                for col in this_data0 if col not in mapped_cols}
                this_data = this_data0[list(unmapped_cols)].rename(columns=unmapped_cols)
                this_data = concat([column_mapper(this_data0), this_data], 
                                    axis=1)
            else:
                this_data = column_mapper(this_data0)
            
            if this_representation.casefold() not in _FROM_FUNCTIONS:

                raise TypeError(' or '.join(list(set(this_representation, this_representation.casefold()))) + 
                                "not a supported string representation. Try one of " + ", ".join(_FROM_FUNCTIONS))
            
            this_converter = partial(convert_string_representation,
                                     input_representation=this_representation.casefold())

            this_data = (this_data
                         .query('compound_collection != "NA"')
                         .assign(library_name=this_library_name,
                                 input_file_format=this_filetype,
                                 input_string_representation=this_representation,
                                 plate_id=lambda x: x['plate_id'].astype(str),
                                 plate_loc=lambda x: x['library_name'].str.cat([x['compound_collection'], x['plate_id'].astype(str), x['well_id'].astype(str)], sep=':'),
                                 canonical_smiles=lambda x: list(this_converter(x[catalog_smiles_column])),
                                 is_valid_smiles=lambda x: [s is not None for s in x['canonical_smiles']]))
                
            report.update({"invalid SMILES": (~this_data['is_valid_smiles']).sum(), 
                           "rows processed": this_data.shape[0]})
            
            if drop_invalid:

                this_data = this_data.query('is_valid_smiles')

            if id_column_name is not None:
                
                this_converter = partial(convert_string_representation, 
                                         output_representation='id',
                                         options=dict(n=id_n_digits,
                                                      prefix=id_prefix))
                this_data = this_data.assign(**{id_column_name: lambda x: list(this_converter(x['canonical_smiles']))})
           
            loaded_dataframes.append(this_data)

    collated_df = concat(loaded_dataframes, axis=0) 

    return report, collated_df


def collate_inventory_from_file(catalog_path: Union[str, TextIO], 
                                root_dir: Optional[str] = None,
                                format: Optional[str] = None,
                                *args, **kwargs) -> DataFrame:
    
    f"""Process a catalog of files containing chemical libraries into a uniform dataframe.

    The catalog table needs to have columns {', '.join(ESSENTIAL_COLUMNS)}:

    - filename is a glob pattern of files to collate
    - file_format is one of {', '.join(FILE_READERS.keys())}
    - smiles_column contains smiles strings

    Other columns are optional and can have any name, but must contain the name or a pattern
    matching a column (for tabular data) or field (for SDF data) in the files 
    of the `filename` column. In the output DataFrame, the named column data will be mapped.

    Optional column contents can be either concatenated or split using the following
    pattern:

    - col1+col2: concatenates the contents of `col1` and `col2`
    - col1;-;1:2 : splits the contents of `col1` on the `-` character, and takes splits 1-2 (0-indexed)

    Parameters
    ----------
    catalog_path : str
        Path to catalog file in XLSX, TSV or CSV format. Requires 
        columns {', '.join(ESSENTIAL_COLUMNS)}.
    format : str, optional
        Format of catalog file. Default: infer from file extension.
    root_dir : str, optional
        Path to look for data files. Default: use directory containing 
        the catalog.

    Returns
    -------
    pd.DataFrame
        Collated chemical data.

    """

    root_dir = root_dir or os.path.dirname(catalog_path)

    data_catalog = read_table(catalog_path, format=format)

    return collate_inventory(catalog=data_catalog, 
                             root_dir=root_dir,
                             *args, **kwargs)


def deduplicate(df: DataFrame,
                column: str = 'smiles', 
                input_representation: str = 'smiles',
                index_columns: Optional[List[str]] = None,
                drop_inchikey: bool = False) -> DataFrame:

    index_columns = index_columns or []
    
    inchikey_converter = partial(convert_string_representation,
                                 input_representation=input_representation,
                                 output_representation='inchikey')
    
    df = df.assign(inchikey=lambda x: inchikey_converter(x[column]))

    structure_columns = [column, 'inchikey']
    df_unique = []

    for (string_rep, inchikey), structure_df in df.groupby(structure_columns):

        collapsed_indexes = {col: [';'.join(sorted(map(str, set(structure_df[col].tolist()))))]
                             for col in structure_df if col in index_columns}
        collapsed_indexes.update({column: [string_rep], 
                                  'inchikey': [inchikey],
                                  'instance_count': [structure_df.shape[0]]})

        df_unique.append(DataFrame(collapsed_indexes))

    df_unique = concat(df_unique, axis=0)

    if drop_inchikey:

        df_unique = df_unique.drop(columns=['inchikey'])

    report = {'starting rows:': df.shape[0], 
              'ending_rows': df_unique.shape[0]}

    return report, df_unique


def deduplicate_file(filename: Union[str, TextIO],
                     format: Optional[str] = None,
                     *args, **kwargs) -> DataFrame:
    
    table = read_table(filename)

    return deduplicate(table, *args, **kwargs)

