"""Gradio demo for schemist."""

from typing import Iterable, List, Union
from io import TextIOWrapper
import os
os.environ["COMMANDLINE_ARGS"] = "--no-gradio-queue"

from carabiner import cast, print_err
from carabiner.pd import read_table
import gradio as gr
import nemony as nm
import numpy as np
import pandas as pd
from rdkit.Chem import Draw, Mol
import schemist as sch
from schemist.converting import (
    _TO_FUNCTIONS,
    _FROM_FUNCTIONS, 
    convert_string_representation, 
    _x2mol,
)
from schemist.tables import converter

def load_input_data(file: TextIOWrapper) -> pd.DataFrame:
    df = read_table(file.name)
    string_cols = list(df.select_dtypes(exclude=[np.number]))
    df = gr.Dataframe(value=df, visible=True)
    return df, gr.Dropdown(choices=string_cols, interactive=True)
    

def _clean_split_input(strings: str) -> List[str]:
    return [s2.strip() for s in strings.split("\n") for s2 in s.split(",")]


def _convert_input(
    strings: str,
    input_representation: str = 'smiles', 
    output_representation: Union[Iterable[str], str] = 'smiles'
) -> List[str]:
    strings = _clean_split_input(strings)
    converted = convert_string_representation(
        strings=strings, 
        input_representation=input_representation, 
        output_representation=output_representation,
    )
    return {key: list(map(str, cast(val, to=list))) for key, val in converted.items()}


def convert_one(
    strings: str,
    input_representation: str = 'smiles', 
    output_representation: Union[Iterable[str], str] = 'smiles'
):

    df = pd.DataFrame({
        input_representation: _clean_split_input(strings),
    })

    return gr.DataFrame(
        convert_file(
            df=df,
            column=input_representation,
            input_representation=input_representation,
            output_representation=output_representation,
        ),
        visible=True
    )


def convert_file(
    df: pd.DataFrame, 
    column: str = 'smiles',
    input_representation: str = 'smiles',
    output_representation: Union[str, Iterable[str]] = 'smiles'
):
    message = f"Converting from {input_representation} to {output_representation}..."
    print_err(message)
    gr.Info(message, duration=3)
    errors, df = converter(
        df=df,
        column=column,
        input_representation=input_representation,
        output_representation=output_representation,
    )
    df = df[
        cast(output_representation, to=list) +
        [col for col in df if col not in output_representation]
    ]
    all_err = sum(err for key, err in errors.items())
    message = (
        f"Converted {df.shape[0]} molecules from "
        f"{input_representation} to {output_representation} "
        f"with {all_err} errors!"
    )
    print_err(message)
    gr.Info(message, duration=5)
    return df


def draw_one(
    strings: Union[Iterable[str], str],
    input_representation: str = 'smiles'
):
    _ids = _convert_input(
        strings, 
        input_representation, 
        ["inchikey", "id"],
    )
    mols = cast(_x2mol(_clean_split_input(strings), input_representation), to=list)
    if isinstance(mols, Mol):
        mols = [mols]
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, len(mols)), 
        subImgSize=(300, 300),
        legends=["\n".join(items) for items in zip(*_ids.values())],
    )


def download_table(
    df: pd.DataFrame
) -> str:
    df_hash = nm.hash(pd.util.hash_pandas_object(df).values)
    filename = f"converted-{df_hash}.csv"
    df.to_csv(filename, index=False)
    return gr.DownloadButton(value=filename, visible=True)

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Chemical string format converter

        """
    )
    with gr.Tab(label="Paste one per line"):
        input_format_single = gr.Dropdown(
            label="Input string format",
            choices=list(_FROM_FUNCTIONS),
            value="smiles",
            interactive=True,
        )
        input_line = gr.Textbox(
            label="Input",
            placeholder="Paste your molecule here, one per line",
            lines=2,
            interactive=True,
            submit_btn=True,
        )
        output_format_single = gr.CheckboxGroup(
            label="Output format",
            choices=list(_TO_FUNCTIONS),
            value=["id", "pubchem_name"],
            interactive=True,
        )
        download_single = gr.DownloadButton(
            label="Download converted data",
            visible=False,
        )
        with gr.Row():
            output_line = gr.DataFrame(
                label="Converted",
                interactive=False,
                visible=False,
            )
            drawing = gr.Image(label="Chemical structures")
        gr.on(
            [
                # go_button.click,
                input_line.submit,
            ],
            fn=convert_one,
            inputs=[
                input_line, 
                input_format_single,
                output_format_single,
            ],
            outputs={
                output_line,
            }
        ).then(
            draw_one,
            inputs=[
                input_line, 
                input_format_single,
            ],
            outputs=drawing,
        ).then(
            download_table,
            inputs=output_line,
            outputs=download_single
        )

    with gr.Tab("Convert a file"):
        input_file = gr.File(
            label="Upload a table of chemical compounds here",
            file_types=[".xlsx", ".csv", ".tsv", ".txt"],
        )
        with gr.Row():
            input_column = gr.Dropdown(
                label="Input column name",
                choices=[],
            )
            input_format = gr.Dropdown(
                label="Input string format",
                choices=list(_FROM_FUNCTIONS),
                value="smiles",
                interactive=True,
            )
        output_format = gr.CheckboxGroup(
            label="Output format",
            choices=list(_TO_FUNCTIONS),
            value=["id", "selfies"],
            interactive=True,
        )
        go_button2 = gr.Button(
            value="Convert molecules!",
        )

        download = gr.DownloadButton(
            label="Download converted data",
            visible=False,
        )
        input_data = gr.Dataframe(
            label="Input data",
            max_height=100,
            visible=False,
            interactive=False,
        )
        
        input_file.upload(
            load_input_data, 
            inputs=[input_file], 
            outputs=[input_data, input_column]
        )
        go_button2.click(
            convert_file,
            inputs=[
                input_data, 
                input_column,
                input_format,
                output_format,
            ],
            outputs={
                input_data,
            }
        ).then(
            download_table,
            inputs=input_data,
            outputs=download
        )

if __name__ == "__main__":
    demo.queue() 
    demo.launch(share=True)

