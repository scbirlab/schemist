"""Tools to facilitate input and output."""

from typing import Any, Callable, List, Optional, TextIO, Tuple, Union

from collections import defaultdict
from functools import partial
from string import printable
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree

from carabiner import print_err
from carabiner.cast import cast
from carabiner.itertools import tenumerate
from carabiner.pd import read_table, write_stream

from pandas import DataFrame, read_excel
from rdkit.Chem import SDMolSupplier

from .converting import _mol2isomeric_canonical_smiles

def _mutate_df_stream(input_file: Union[str, TextIO], 
                      output_file: Union[str, TextIO], 
                      function: Callable[[DataFrame], Tuple[Any, DataFrame]],
                      file_format: Optional[str] = None, 
                      chunksize: int = 1000) -> List[Any]:
    
    carries = []

    for i, chunk in tenumerate(read_table(input_file, 
                                          format=file_format,
                                          progress=False,
                                          chunksize=chunksize)):
        
        result = function(chunk)

        try:
            carry, df = result
        except ValueError:
            df = result
            carry = 0
        
        write_stream(df, 
                     output=output_file,
                     format=file_format,
                     header=i == 0,
                     mode='w' if i == 0 else 'a')
        
        carries.append(carry)

    return carries


def read_weird_xml(filename: Union[str, TextIO], 
                   header: bool = True, 
                   namespace: str = '{urn:schemas-microsoft-com:office:spreadsheet}') -> DataFrame:

    """

    """

    with cast(filename, TextIO, mode='r') as f:

        xml_string = ''.join(filter(printable.__contains__, f.read()))

    try:

        root = ElementTree.fromstring(xml_string)

    except Exception as e:

        print_err('\n!!! ' + xml_string.split('\n')[1184][377:380])

        raise e
    
    for i, row in enumerate(root.iter(f'{namespace}Row') ):

        this_row = [datum.text for datum in row.iter(f'{namespace}Data')] 

        if i == 0:

            if header:

                heading = this_row
                df = {colname: [] for colname in heading}

            else:

                heading = [f'X{j}' for j, _ in enumerate(this_row)]
                df = {colname: [datum] for colname, datum in zip(heading, this_row)}

        else:

            for colname, datum in zip(heading, this_row):

                df[colname].append(datum)

    return DataFrame(df)


def read_sdf(filename: Union[str, TextIO]):

    """

    """

    filename = cast(filename, str)

    with open(filename, 'r', errors='replace') as f:
        with NamedTemporaryFile("w") as o:

            o.write(f.read())
            o.seek(0)

            df = defaultdict(list)

            for i, mol in enumerate(SDMolSupplier(o.name)):

                if mol is None: 

                    continue
                
                propdict = mol.GetPropsAsDict()
                propdict['SMILES'] = _mol2isomeric_canonical_smiles(mol)

                for colname in propdict:

                    df[colname].append(propdict[colname])

                for colname in df:

                    if colname not in propdict:

                        df[colname].append(None)

    col_lengths = {col: len(val) for col, val in df.items()}

    if len(set(col_lengths.values())) > 1:

        raise ValueError(f"Column lengths not all the same:\n\t" +
                         '\n\t'.join(f"{key}:{val}" for key, val in col_lengths.items()))

    return DataFrame(df)


FILE_READERS = {
    'bad_xml': read_weird_xml,
    'xlsx': partial(read_excel, engine='openpyxl'),
    'sdf': read_sdf
}
