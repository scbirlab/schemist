"""Command-line interface for schemist."""

from typing import Any, Dict, List, Optional

from argparse import FileType, Namespace
from collections import Counter, defaultdict
from functools import partial
import os
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

from carabiner import pprint_dict, upper_and_lower
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp
from carabiner.itertools import tenumerate
from carabiner.pd import get_formats, write_stream

from . import __version__
from .collating import collate_inventory, deduplicate_file
from .converting import _TO_FUNCTIONS, _FROM_FUNCTIONS
from .generating import AA, REACTIONS
from .io import _mutate_df_stream
from .tables import (converter, cleaner, featurizer, assign_groups, 
                     _assign_splits, splitter, _peptide_table, reactor)
from .splitting import _SPLITTERS, _GROUPED_SPLITTERS

def _option_parser(x: Optional[List[str]]) -> Dict[str, Any]:

    options = {}

    try:
        for opt in x:

            try:
                key, value = opt.split('=')
            except ValueError:
                raise ValueError(f"Option {opt} is misformatted. It should be in the format keyword=value.")
            
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            options[key] = value

    except TypeError:
        
        pass

    return options


def _sum_tally(tallies: Counter, 
               message: str = "Error counts",
               use_length: bool = False):

    total_tally = Counter()
    
    for tally in tallies:

        if use_length:
            total_tally.update({key: len(value) for key, value in tally.items()})
        else:
            total_tally.update(tally)

    if len(tallies) == 0:
        raise ValueError(f"Nothing generated!")

    pprint_dict(total_tally, message=message)

    return total_tally


@clicommand(message="Cleaning file with the following parameters")
def _clean(args: Namespace) -> None:
    
    error_tallies = _mutate_df_stream(input_file=args.input, 
                                      output_file=args.output, 
                                      function=partial(cleaner, 
                                                       column=args.column, 
                                                       input_representation=args.representation, 
                                                       prefix=args.prefix),
                                      file_format=args.format)
    
    _sum_tally(error_tallies)
    
    return None


@clicommand(message="Converting between string representations with the following parameters")
def _convert(args: Namespace) -> None:
    
    options = _option_parser(args.options)
    
    error_tallies = _mutate_df_stream(input_file=args.input, 
                                      output_file=args.output, 
                                      function=partial(converter, 
                                                       column=args.column, 
                                                       input_representation=args.representation, 
                                                       output_representation=args.to,
                                                       prefix=args.prefix,
                                                       options=options),
                                      file_format=args.format)
    
    _sum_tally(error_tallies)

    return None


@clicommand(message="Adding features to files with the following parameters")
def _featurize(args: Namespace) -> None:
    
    error_tallies = _mutate_df_stream(input_file=args.input, 
                                      output_file=args.output, 
                                      function=partial(featurizer,
                                                       feature_type=args.feature,
                                                       column=args.column, 
                                                       ids=args.id,
                                                       input_representation=args.representation, 
                                                       prefix=args.prefix),
                                      file_format=args.format)
    
    _sum_tally(error_tallies)

    return None


@clicommand(message="Splitting table with the following parameters")
def _split(args: Namespace) -> None:
    
    split_type = args.type.casefold()

    if split_type in _GROUPED_SPLITTERS:

        chunk_processor, aggregator = _GROUPED_SPLITTERS[split_type]

        with TemporaryDirectory() as dir:

            with NamedTemporaryFile("w", dir=dir, delete=False) as f:

                group_idxs = _mutate_df_stream(input_file=args.input, 
                                                output_file=f, 
                                                function=partial(assign_groups,
                                                                grouper=chunk_processor,
                                                                group_name=split_type,
                                                                column=args.column, 
                                                                input_representation=args.representation),
                                                file_format=args.format)
                f.close()
                new_group_idx = defaultdict(list)

                totals = 0
                for group_idx in group_idxs:
                    these_totals = 0
                    for key, value in group_idx.items():
                        these_totals += len(value)
                        new_group_idx[key] += [idx + totals for idx in value]
                    totals += these_totals

                group_idx = aggregator(new_group_idx, 
                                    train=args.train, 
                                    test=args.test)
                
                split_tallies = _mutate_df_stream(input_file=f.name, 
                                                  output_file=args.output, 
                                                  function=partial(_assign_splits,
                                                                  split_idx=group_idx,
                                                                  use_df_index=True),
                                                  file_format=args.format)
                if os.path.exists(f.name):
                    os.remove(f.name)
        
    else:

        split_tallies = _mutate_df_stream(input_file=args.input, 
                                          output_file=args.output, 
                                          function=partial(splitter,
                                                           split_type=args.type,
                                                           column=args.column, 
                                                           input_representation=args.representation, 
                                                           train=args.train, 
                                                           test=args.test,
                                                           set_seed=args.seed),
                                          file_format=args.format)
    
    _sum_tally(split_tallies, 
               message="Split counts")

    return None


@clicommand(message="Collating files with the following parameters")
def _collate(args: Namespace) -> None:
    
    root_dir = args.data_dir or '.'
    
    error_tallies = _mutate_df_stream(input_file=args.input, 
                                      output_file=args.output, 
                                      function=partial(collate_inventory, 
                                                       root_dir=root_dir,
                                                       drop_unmapped=not args.keep_extra_columns,
                                                       catalog_smiles_column=args.column,
                                                       id_column_name=args.id_column,
                                                       id_n_digits=args.digits,
                                                       id_prefix=args.prefix),
                                      file_format=args.format)
    
    _sum_tally(error_tallies,
               message="Collated chemicals:")

    return None


@clicommand(message="Deduplicating chemical structures with the following parameters")
def _dedup(args: Namespace) -> None:

    report, deduped_df = deduplicate_file(args.input,
                                      format=args.format,
                                      column=args.column, 
                                      input_representation=args.representation,
                                      index_columns=args.indexes)
    
    if args.prefix is not None and 'inchikey' in deduped_df:
        deduped_df = deduped_df.rename(columns={'inchikey': f'{args.prefix}inchikey'})
    
    write_stream(deduped_df, 
                 output=args.output,
                 format=args.format)
    
    pprint_dict(report, message="Finished deduplicating:")

    return None


@clicommand(message="Enumerating peptides with the following parameters")
def _enum(args: Namespace) -> None:
    
    tables = _peptide_table(max_length=args.max_length,
                            min_length=args.min_length,
                            n=args.number,
                            indexes=args.slice,
                            set_seed=args.seed,
                            prefix=args.prefix,
                            suffix=args.suffix,
                            d_aa_only=args.d_aa_only,
                            include_d_aa=args.include_d_aa, 
                            generator=True)

    dAA_use = any(aa.islower() for aa in args.prefix + args.suffix)
    dAA_use = dAA_use or args.include_d_aa or args.d_aa_only
    
    tallies, error_tallies = [], []
    options = _option_parser(args.options)
    _converter = partial(converter, 
                         column='peptide_sequence', 
                         input_representation='minihelm' if dAA_use else 'aa_seq',  ## affects performance
                         output_representation=args.to,
                         options=options)
    
    for i, table in tenumerate(tables):

        _err_tally, df = _converter(table)

        tallies.append({"Number of peptides": df.shape[0]})
        error_tallies.append(_err_tally)

        write_stream(df, 
                     output=args.output,
                     format=args.format,
                     mode='w' if i == 0  else 'a',
                     header=i == 0)

    _sum_tally(tallies,
               message="Enumerated peptides")
    _sum_tally(error_tallies,
               message="Conversion errors")

    return None


@clicommand(message="Reacting peptides with the following parameters")
def _react(args: Namespace) -> None:

    error_tallies = _mutate_df_stream(input_file=args.input, 
                                      output_file=args.output, 
                                      function=partial(reactor, 
                                                       column=args.column, 
                                                       input_representation=args.representation,
                                                       reaction=args.reaction,
                                                       product_name=args.name),
                                      file_format=args.format)
    
    _sum_tally(error_tallies)

    return None


def main() -> None:

    inputs = CLIOption('input', 
                       default=sys.stdin,
                       type=FileType('r'), 
                       nargs='?',
                       help='Input columnar Excel, CSV or TSV file. Default: STDIN.')
    representation = CLIOption('--representation', '-r', 
                       type=str,
                       default='SMILES',
                       choices=upper_and_lower(_FROM_FUNCTIONS),
                       help='Chemical representation to use for input. ')
    column = CLIOption('--column', '-c', 
                       default='smiles',
                       type=str,
                       help='Column to use as input string representation. ')
    prefix = CLIOption('--prefix', '-p', 
                       default=None,
                       type=str,
                       help='Prefix to add to new column name. Default: no prefix')
    to = CLIOption('--to', '-2', 
                       type=str,
                       default='SMILES',
                       nargs='*',
                       choices=upper_and_lower(_TO_FUNCTIONS),
                       help='Format to convert to.')
    options = CLIOption('--options', '-x', 
                       type=str,
                       default=None,
                       nargs='*',
                       help='Options to pass to converter, in the format '
                           '"keyword1=value1 keyword2=value2"')
    output = CLIOption('--output', '-o', 
                       type=FileType('w'),
                       default=sys.stdout,
                       help='Output file. Default: STDOUT')                
    formatting = CLIOption('--format', '-f', 
                           type=str,
                           default=None,
                           choices=upper_and_lower(get_formats()),
                           help='Override file extensions for input and output. '
                                'Default: infer from file extension.')

    ## featurize
    id_feat = CLIOption('--id', '-i', 
                        type=str,
                        default=None,
                        nargs='*',
                        help='Columns to retain in output table. Default: use all')
    feature = CLIOption('--feature', '-t', 
                           type=str,
                           default='2d',
                           choices=['2d', 'fp'],  ## TODO: implement 3d
                           help='Which feature type to generate.')

    ## split
    type_ = CLIOption('--type', '-t', 
                       type=str,
                       default='random',
                       choices=upper_and_lower(_SPLITTERS),
                       help='Which split type to use.')
    train = CLIOption('--train', '-a', 
                       type=float,
                       default=1.,
                       help='Proportion of data to use for training. ')
    test = CLIOption('--test', '-b', 
                       type=float,
                       default=0.,
                       help='Proportion of data to use for testing. ')

    ## collate
    data_dir = CLIOption('--data-dir', '-d', 
                         type=str,
                         default=None,
                         help='Directory containing data files. '
                              'Default: current directory')
    id_column = CLIOption('--id-column', '-s', 
                         default=None,
                         type=str,
                         help='If provided, add a structure ID column with this name. '
                              'Default: don\'t add structure IDs')
    prefix_collate = CLIOption('--prefix', '-p', 
                         default='ID-',
                         type=str,
                         help='Prefix to add to structure IDs. '
                              'Default: no prefix')
    digits = CLIOption('--digits', '-n', 
                         default=8,
                         type=int,
                         help='Number of digits in structure IDs. ')
    keep_extra_columns = CLIOption('--keep-extra-columns', '-x', 
                         action='store_true',
                         help='Whether to keep columns not mentioned in the catalog. '
                              'Default: drop extra columns.')
    keep_invalid_smiles = CLIOption('--keep-invalid-smiles', '-y', 
                         action='store_true',
                         help='Whether to keep rows with invalid SMILES. '
                              'Default: drop invalid rows.')

    ## dedup
    indexes = CLIOption('--indexes', '-x', 
                        type=str,
                        default=None,
                        nargs='*',
                         help='Columns to retain and collapse (if multiple values per unique structure). '
                              'Default: retain no other columns than structure and InchiKey.')
    drop_inchikey = CLIOption('--drop-inchikey', '-d', 
                         action='store_true',
                         help='Whether to drop the calculated InchiKey column. '
                              'Default: keep InchiKey.')
    
    ### enum 
    max_length = CLIOption('--max-length', '-l', 
                           type=int,
                           help='Maximum length of enumerated peptide. '
                                'Required.')
    min_length = CLIOption('--min-length', '-m', 
                      type=int,
                      default=None,
                      help='Minimum length of enumerated peptide. '
                           'Default: same as maximum, i.e. all peptides same length.')
    number_to_gen = CLIOption('--number', '-n', 
                              type=float,
                              default=None,
                              help='Number of peptides to sample from all possible '
                                   'within the constraints. If less than 1, sample '
                                   'that fraction of all possible. If greater than 1, '
                                   'sample that number. '
                                   'Default: return all peptides.')
    slicer = CLIOption('--slice', '-z', 
                       type=str,
                       default=None,
                       nargs='*',
                       help='Subset of (possibly sampled) population to return, in the format <stop> '
                            'or <start> <stop> [<step>]. If "x" is used for <stop>, then it runs to the end. '
                            'For example, 1000 gives the first 1000, 2 600 gives items 2-600, and '
                            '3 500 2 gives every other from 3 to 500. Default: return all.')
    alphabet = CLIOption('--alphabet', '-b', 
                      type=str,
                      default=''.join(AA),
                      help='Alphabet to use in sampling.')
    suffix = CLIOption('--suffix', '-s', 
                      type=str,
                      default='',
                      help='Sequence to add to end. Lowercase for D-amino acids. '
                           'Default: no suffix.')
    set_seed = CLIOption('--seed', '-e', 
                      type=int,
                      default=None,
                      help='Seed to use for reproducible randomness. '
                           'Default: don\'t enable reproducibility.')
    d_aa_only = CLIOption('--d-aa-only', '-a', 
                      action='store_true',
                      help='Whether to only use D-amino acids. '
                           'Default: don\'t include.')
    include_d_aa = CLIOption('--include-d-aa', '-y', 
                      action='store_true',
                      help='Whether to include D-amino acids in enumeration. '
                           'Default: don\'t include.')

    ## reaction
    name = CLIOption('--name', '-n', 
                     type=str,
                     default=None,
                     help='Name of column for product. '
                          'Default: same as reaction name.')
    reaction_opt = CLIOption('--reaction', '-x', 
                             type=str,
                             nargs='*',
                             choices=list(REACTIONS),
                             default='N_to_C_cyclization',
                             help='Reaction(s) to apply.')

    clean = CLICommand('clean', 
                       description='Clean and normalize SMILES column of a table.',
                       main=_clean,
                       options=[output, formatting, inputs, representation, column, prefix])
    convert = CLICommand('convert', 
                         description='Convert between string representations of chemical structures.',
                         main=_convert,
                         options=[output, formatting, inputs, representation, column, prefix, to, options])
    featurize = CLICommand('featurize', 
                         description='Convert between string representations of chemical structures.',
                         main=_featurize,
                         options=[output, formatting, inputs, representation, column, prefix,
                                  id_feat, feature])
    collate = CLICommand('collate', 
                         description='Collect disparate tables or SDF files of libraries into a single table.',
                         main=_collate,
                         options=[output, formatting, inputs, representation,
                                  data_dir, column.replace(default='input_smiles'), id_column, prefix_collate,
                                  digits, keep_extra_columns, keep_invalid_smiles])
    dedup = CLICommand('dedup', 
                         description='Deduplicate chemical structures and retain references.',
                         main=_dedup,
                         options=[output, formatting, inputs, representation, column, prefix,
                                  indexes, drop_inchikey])
    enum = CLICommand('enumerate', 
                      description='Enumerate bio-chemical structures within length and sequence constraints.',
                      main=_enum,
                      options=[output, formatting, to, options,
                               alphabet, max_length, min_length, number_to_gen,
                               slicer, set_seed,
                               prefix.replace(default='',  
                                              help='Sequence to prepend. Lowercase for D-amino acids. '
                                                   'Default: no prefix.'), 
                               suffix, 
                               type_.replace(default='aa',
                                             choices=['aa'],
                                             help='Type of bio sequence to enumerate. '
                                                  'Default: %(default)s.'), 
                               d_aa_only, include_d_aa])
    reaction = CLICommand('react', 
                         description='React compounds in silico in indicated columns using a named reaction.',
                         main=_react,
                         options=[output, formatting, inputs, representation, column, name,
                                  reaction_opt])
    split = CLICommand('split', 
                         description='Split table based on chosen algorithm, optionally taking account of chemical structure during splits.',
                         main=_split,
                         options=[output, formatting, inputs, representation, column, prefix,
                                  type_, train, test, set_seed])

    app = CLIApp("schemist",
                 version=__version__,
                 description="Tools for cleaning, collating, and augmenting chemical datasets.",
                 commands=[clean, convert, featurize, collate, dedup, enum, reaction, split])

    app.run()

    return None


if __name__ == "__main__":

    main()