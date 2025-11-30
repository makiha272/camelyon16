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
    return plt, sf


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


if __name__ == "__main__":
    app.run()
