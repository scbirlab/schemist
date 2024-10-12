# ⬢⬢⬢ schemist

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/schemist/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/schemist)
![PyPI](https://img.shields.io/pypi/v/schemist)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/scbirlab/chem-converter)

Cleaning, collating, and augmenting chemical datasets.

- [Installation](#installation)
- [Command-line usage](#command-line-usage)
- [Python API](#python-api)
- [Documentation](#documentation)

## Installation

### The easy way

Install the pre-compiled version from PyPI:

```bash
pip install schemist
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
pip install -e .
```

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

**schemist** can be imported into Python to help make custom analyses.

```python
>>> import schemist as sch
```

## Documentation

Full API documentation is at [ReadTheDocs](https://schemist.readthedocs.org).