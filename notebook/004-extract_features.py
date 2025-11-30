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
    return mo, plt, sf


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    return (P,)


@app.cell
def _(P):
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return (dataset,)


@app.cell
def _(dataset):
    dataset.extract_tiles()
    return


@app.cell
def _(sf):
    tfr = sf.TFRecord('/workspace/slideflow_project/tfrecords/224px_10x/normal_015.tfrecords')
    return (tfr,)


@app.cell
def _(plt, sf, tfr):
    plt.imshow(sf.io.decode_image(tfr[1]['image_raw']))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell
def _(sf):
    hp = sf.ModelParams(
        tile_px=256,
        tile_um='10x',
        epochs=list(range(1, 6)), #5エポック
        model='resnet18',
        loss='CrossEntropy',
        optimizer='Adam',
        learning_rate=0.0001,
        batch_size=128,
        early_stop=True,
        early_stop_patience=3,
        dropout=0.5,
        # augment='xyrn',
        # normalizer='macenko'
    )
    return (hp,)


@app.cell
def _(P, hp):
    results = P.train(
      outcomes="category",
      params=hp,
      filters={"category": ["tumor", "normal"]},
      val_strategy="fixed",
      val_fraction=0.2,
      use_tensorboard=True
    )
    return


@app.cell
def _(P):
    dfs = P.predict(
      model="/workspace/slideflow_project/models/00005-category-HP0/category-HP0_epoch5.zip",
      filters={"category": ["test"]}
    )
    return (dfs,)


@app.cell
def _(dfs):
    dfs
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
