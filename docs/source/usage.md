# Usage

**schemist** has a variety of utilities which can be used through the command-line or the [Python API](#python-api).

## Command-line usage

**schemist**  provides command-line utlities. The list of commands can be checked like so:

```bash
$ schemist --help
usage: schemist [-h] [--version] {clean,convert,featurize,collate,dedup,enumerate,react,split} ...

Tools for cleaning, collating, and augmenting chemical datasets.

options:
  -h, --help            show this help message and exit
  --version, -v         show program's version number and exit

Sub-commands:
  {clean,convert,featurize,collate,dedup,enumerate,react,split}
                        Use these commands to specify the tool you want to use.
    clean               Clean and normalize SMILES column of a table.
    convert             Convert between string representations of chemical structures.
    featurize           Convert between string representations of chemical structures.
    collate             Collect disparate tables or SDF files of libraries into a single table.
    dedup               Deduplicate chemical structures and retain references.
    enumerate           Enumerate bio-chemical structures within length and sequence constraints.
    react               React compounds in silico in indicated columns using a named reaction.
    split               Split table based on chosen algorithm, optionally taking account of chemical structure during splits.
```

Each command is designed to work on large data files in a streaming fashion, so that the entire file is not held in memory at once. One caveat is that the scaffold-based splits are very slow with tables of millions of rows.

All commands (except `collate`) take from the input table a named column with a SMILES, SELFIES, amino-acid sequence, HELM, or InChI representation of compounds.

The tools complete specific tasks which 
can be easily composed into analysis pipelines, because the TSV table output goes to
`stdout` by default so they can be piped from one tool to another.

To get help for a specific command, do

```bash
schemist <command> --help
```

For the Python API, [see below](#python-api).


## Python API

You can access the underlying functions of `schemist` to help custom analyses or develop other tools.

```python
>>> import schemist as sch
```