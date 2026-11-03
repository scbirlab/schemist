# ⬢⬢⬢ schemist

[![Python package](https://github.com/scbirlab/schemist/actions/workflows/python-package.yml/badge.svg)](https://github.com/scbirlab/schemist/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/schemist)](https://pypi.org/project/schemist/)
[![Python versions](https://img.shields.io/pypi/pyversions/schemist)](https://pypi.org/project/schemist/)
[![Documentation](https://readthedocs.org/projects/schemist/badge/?version=latest)](https://schemist.readthedocs.io/)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/scbirlab/chem-converter)

**schemist** is a Python toolkit for cleaning, converting, featurizing, and organizing chemical datasets.

Built around RDKit and pandas, it provides composable command-line utilities and a Python API for cheminformatics workflows, including molecular descriptors, fingerprints, chemical library management, peptide enumeration, and structure-aware dataset splitting.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Command-line interface](#command-line-interface)
  - [Cleaning and conversion](#cleaning-and-conversion)
  - [Molecular features](#molecular-features)
  - [Dataset splitting](#dataset-splitting)
  - [Library collation and deduplication](#library-collation-and-deduplication)
  - [Peptides and reactions](#peptides-and-reactions)
- [Python API](#python-api)
- [Data handling](#data-handling)
- [Documentation](#documentation)
- [License](#license)

## Quick start

Install the package:

```bash
pip install schemist
```

Suppose `compounds.csv` contains a column named `smiles`:

```csv
compound_id,smiles
1,CCO
2,CC(=O)O
3,c1ccccc1
```

Standardize the structures, calculate molecular descriptors, and assign a reproducible scaffold-based train/test split:

```bash
schemist clean compounds.csv \
  | schemist featurize --feature 2d \
  | schemist split --type scaffold --train 0.8 --test 0.2 \
  > processed.tsv
```

The resulting table contains the original compound information, normalized SMILES, molecular descriptors, and dataset assignments.

Commands can be chained through standard Unix pipes or used independently.

For 3D descriptors, generate multiple conformers per compound:

```bash
schemist featurize compounds.csv \
  --feature 3d \
  --conformers 10 \
  --minimize \
  --threads 4 \
  --output descriptors.csv
```

This uses RDKit conformer embedding, optional MMFF94 minimization, and aggregation of 3D descriptors across conformers.

## Installation

### From PyPI

```bash
pip install schemist
```

### Development version

```bash
git clone https://github.com/scbirlab/schemist.git
cd schemist
git checkout dev
pip install -e ".[dev]"
```

The development branch targets modern RDKit releases and Python 3.11+.

## Command-line interface

```bash
schemist --help
schemist <command> --help
```

| Command | Description |
|---------|-------------|
| `clean` | Sanitize and canonicalize chemical structures |
| `convert` | Convert structures between molecular representations |
| `featurize` | Generate 2D descriptors, 3D descriptors, or fingerprints |
| `split` | Create random or scaffold-based dataset splits |
| `collate` | Merge heterogeneous chemical libraries using a catalog |
| `dedup` | Identify and collapse duplicate structures |
| `enumerate` | Enumerate or sample peptide sequences |
| `react` | Apply predefined chemical reactions |

### Cleaning and conversion

Canonicalize SMILES:

```bash
schemist clean compounds.csv -o cleaned.csv
```

Convert SMILES to alternative representations:

```bash
schemist convert compounds.csv \
  --to smiles inchikey scaffold selfies \
  -o converted.csv
```

Supported input representations include:

- SMILES
- SELFIES
- InChI
- Amino-acid sequences (`aa_seq`)
- HELM and mini-HELM peptide notation

Available output representations include canonical and randomized SMILES, SELFIES, InChI, InChIKey, Murcko scaffolds, molecular hashes, mnemonic identifiers, and generated compound IDs.

The converter also supports physicochemical properties such as molecular weight, cLogP, TPSA, and partial-charge extrema, as well as compound-name and identifier lookups through PubChem and the NCI Chemical Identifier Resolver.

### Molecular features

The `featurize` command provides three feature families:

| Feature | Flag | Description |
|---------|------|-------------|
| 2D descriptors | `--feature 2d` | RDKit descriptors via descriptastorus |
| 3D descriptors | `--feature 3d` | RDKit shape and geometry descriptors |
| Fingerprints | `--feature fp` | Morgan circular fingerprints |

**2D descriptors**

```bash
schemist featurize compounds.csv \
  --feature 2d \
  -o features-2d.csv
```

**Morgan fingerprints**

```bash
schemist featurize compounds.csv \
  --feature fp \
  -o fingerprints.csv
```

**3D descriptors**

```bash
schemist featurize compounds.csv \
  --feature 3d \
  --conformers 20 \
  --minimize \
  --threads 4 \
  -o features-3d.csv
```

3D descriptors are calculated using RDKit ETKDGv4 conformer embedding. Multiple conformers can be generated for each molecule, with descriptor values averaged across conformers.

| Option | Default | Description |
|--------|---------|-------------|
| `--conformers`, `-k` | `1` | Number of conformers per molecule |
| `--threads`, `-j` | `1` | Parallel threads; `0` uses all available |
| `--minimize` | Off | Minimize conformers using MMFF94 |
| `--zscore` | Off | Divide descriptor means by their population standard deviation across conformers |

The `--zscore` option requires multiple conformers to have an effect. It is an experimental normalization option and should be used cautiously for descriptors with little or no conformational variance.

Feature tables include metadata identifying feature type and calculation validity.

### Dataset splitting

Create reproducible random splits:

```bash
schemist split compounds.csv \
  --type random \
  --train 0.8 \
  --test 0.2 \
  --seed 42 \
  -o splits.csv
```

Create scaffold-disjoint splits:

```bash
schemist split compounds.csv \
  --type scaffold \
  --train 0.8 \
  --test 0.2 \
  -o scaffold-splits.csv
```

Two splitting strategies are available:

- **Random:** assign compounds to partitions using random sampling.
- **Scaffold:** group compounds by their Bemis–Murcko scaffolds before assigning partitions, reducing scaffold overlap between partitions.

The remaining fraction, if any, is assigned to validation.

Scaffold splitting requires global grouping of structures and can be slower for very large datasets.

### Library collation and deduplication

Collate chemical libraries from multiple tabular or structure files using a catalog:

```bash
schemist collate inventory.csv \
  --data-dir libraries/ \
  -o combined.csv
```

The inventory specifies source files, their formats, structural representations, and column mappings.

Additional options support structure identifier generation and retention of unmapped columns.

Deduplicate a compound table:

```bash
schemist dedup compounds.csv \
  --column smiles \
  --indexes compound_id \
  -o unique.csv
```

Deduplication uses molecular identifiers to group structures, reports occurrence counts, and can retain collapsed references to the original records.

### Peptides and reactions

Enumerate peptides of a specified length:

```bash
schemist enumerate \
  --max-length 5 \
  --number 1000 \
  --seed 42 \
  -o peptides.tsv
```

Enumeration supports length constraints, fixed sequence prefixes and suffixes, sampling, and D-amino acids.

Apply a predefined reaction to a table of molecular structures:

```bash
schemist react peptides.csv \
  --reaction N_to_C_cyclization \
  -o products.csv
```

The reaction engine uses RDKit reaction SMARTS. Currently, the predefined reaction set includes N-to-C cyclization.

## Python API

The underlying functionality is available directly in Python.

### Convert molecular representations

```python
from schemist.converting import convert_string_representation

smiles = ["CCO", "CC(=O)O", "c1ccccc1"]

identifiers = convert_string_representation(
    smiles,
    output_representation=["smiles", "inchikey", "scaffold"],
)
```

### Generate molecular descriptors

```python
from schemist.features import (
    calculate_2d_features,
    calculate_3d_features,
    calculate_fingerprints,
)

smiles = ["CCO", "CC(=O)O", "c1ccccc1"]

descriptors_2d = calculate_2d_features(
    strings=smiles,
    return_dataframe=True,
)

descriptors_3d = calculate_3d_features(
    strings=smiles,
    n_conformers=10,
    minimize=True,
    n_threads=4,
    return_dataframe=True,
)

fingerprints = calculate_fingerprints(
    strings=smiles,
    return_dataframe=True,
)
```

For combined feature generation, use `schemist.features.calculate_feature`.

The Python API also exposes functions for structure cleaning, dataset splitting, chemical library collation, peptide generation, and in-silico reactions.

## Data handling

**schemist** is designed for processing tabular chemical datasets.

- **Input:** CSV, TSV, Excel, and supported structured chemical-library files.
- **Output:** Tabular results, with TSV written to standard output by default.
- **Streaming:** Most CLI operations process data in chunks to limit memory consumption.
- **Composition:** Standard input/output allows commands to be connected in shell pipelines.
- **Error handling:** Invalid or unprocessable structures are tracked, with validity indicators or error counts where supported.

Operations requiring global information, such as structure deduplication and scaffold grouping, may require additional memory or intermediate storage.

## Documentation

- [API documentation](https://schemist.readthedocs.io/)
- [Source code](https://github.com/scbirlab/schemist)
- [Issue tracker](https://github.com/scbirlab/schemist/issues)
- [Interactive chemical converter](https://huggingface.co/spaces/scbirlab/chem-converter)

## License

MIT License. See [LICENSE](LICENSE).
