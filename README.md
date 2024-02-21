# ⬢⬢⬢ schemist

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/schemist/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/schemist)
![PyPI](https://img.shields.io/pypi/v/schemist)

Cleaning, collating, and augmenting chemical datasets.

- [Installation](#installation)
- [Command-line usage](#command-line-usage)
    - [Example](#example)
    - [Other commands](#other-commands)
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

**schemist**  provides command-line utlities to ... The tools complete specific tasks which 
can be easily composed into analysis pipelines, because the TSV table output goes to
`stdout` by default so they can be piped from one tool to another.

To get a list of commands (tools), do

```bash
schemist --help
```

And to get help for a specific command, do

```bash
schemist <command> --help
```

For the Python API, [see below](#python-api).

## Example


## Other commands



## Python API

**schemist** can be imported into Python to help make custom analyses.

```python
>>> import schemist as sch
```



## Documentation

Full API documentation is at [ReadTheDocs](https://schemist.readthedocs.org).