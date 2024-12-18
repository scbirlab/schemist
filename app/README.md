---
title: Chemical string format converter
emoji: ⚗️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.2"
app_file: app.py
pinned: false
short_description: Trivial batch interconversion of 1D chemical formats.
---
# Chemical string format converter

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/scbirlab/chem-converter)

Trivial batch interconversion of 1D chemical formats.

Frontend for [schemist](https://github.com/scbirlab/schemist) to allow interconversion from:

- SMILES
- SELFIES
- Amino acid sequences
- HELM

to...

- Strucure image
- SMILES
- SELFIES
- InChI
- InChIKey
- Name 
- Murcko scaffold
- Crippen LogP
- TPSA
- Molecular weight
- Charge

... and several others!