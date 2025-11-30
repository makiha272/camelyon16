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
    return (sf,)


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return P, dataset


@app.cell
def _(P, dataset, sf):
    resnet = sf.build_feature_extractor(name='resnet50_imagenet', center_crop=True, tile_px=224)

    P.generate_feature_bags(
        model=resnet, 
        dataset=dataset,
        outdir='/workspace/slideflow_project/bags'
    )
    return


if __name__ == "__main__":
    app.run()
