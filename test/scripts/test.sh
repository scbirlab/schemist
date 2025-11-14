#!/usr/bin/env bash
set -euox pipefail

mkdir -p test/outputs

cat test/data/AmpC_screen_table_2k.csv \
| schemist convert \
    -f CSV \
    -2 id hash smiles inchikey clogp mwt scaffold \
    -o test/outputs/test.csv

head -n201 test/outputs/test.csv \
| schemist convert \
    -f CSV \
    -2 pubchem_id pubchem_name \
    -o test/outputs/test-pubchem.csv

schemist featurize test/outputs/test.csv \
    --feature 2d \
    -o test/outputs/test-features.csv

schemist featurize test/outputs/test.csv \
    --feature 3d \
    -o test/outputs/test-features-3d.csv

schemist featurize test/outputs/test.csv \
    --feature fp \
    -o test/outputs/test-features-fp.csv
