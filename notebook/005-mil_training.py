# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import slideflow as sf
    import matplotlib.pyplot as plt
    import pandas as pd
    return pd, sf


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return (dataset,)


@app.cell
def _(dataset):
    train_dataset, valid_dataset = dataset.split(
        model_type='classification', # Categorical labels
        labels='category',            # Label to balance between datasets
        val_strategy='fixed',          # Fixed split strategy
        val_fraction=0.4,             # 20% of data for testing
        splits='/workspace/slideflow_project/splits.json'         # Where to save/load crossfold splits
    )
    return train_dataset, valid_dataset


@app.cell
def _():
    from slideflow.mil import mil_config
    config = mil_config(
        model='attention_mil',
        lr=1e-4,
        batch_size=2,
        epochs=10,
        fit_one_cycle=True
    )
    return (config,)


@app.cell
def _(config, train_dataset, valid_dataset):
    from slideflow.mil import train_mil
    train_mil(
        config,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        outcomes='category',
        bags='/workspace/slideflow_project/bags',
        outdir='/workspace/slideflow_project/models'
    )
    return


@app.cell
def _(pd):
    df = pd.read_parquet("/workspace/slideflow_project/models/00002-attention_mil-category/predictions.parquet")
    df
    return


if __name__ == "__main__":
    app.run()
