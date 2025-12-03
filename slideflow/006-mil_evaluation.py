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
def _(P, dataset):
    df = P.evaluate_mil(
        model='/workspace/slideflow_project/models/00008-attention_mil-category',
        outcomes='category',
        dataset=dataset,
        bags='/workspace/slideflow_project/bags',
        attention_heatmaps=True
    )
    return


@app.cell
def _(dataset, sf):
    import numpy as np
    from pathlib import Path
    from slideflow.mil.eval import generate_attention_heatmaps

    model_dir = Path('/workspace/slideflow_project/mil_eval/00002-attention_mil')
    att_dir = model_dir / 'attention'

    # Find bags used by the model and build y_att in the same order
    bags = dataset.get_bags('/workspace/slideflow_project/bags')
    bag_map = {sf.util.path_to_name(b): b for b in bags}

    slides_to_process = []
    for f in sorted(att_dir.iterdir()):
        slidename = f.name.split('_att')[0]
        if slidename in bag_map:
            slides_to_process.append(slidename)

    bags_to_use = [bag_map[s] for s in slides_to_process]
    y_att = []
    for s in slides_to_process:
        f = att_dir / f'{s}_att.npz'
        if not f.exists():
            f = att_dir / f'{s}_att.npy'
        data = np.load(f)
        if hasattr(data, 'files') and len(data.files):
            arr = data[data.files[0]]
        else:
            arr = data
        y_att.append(arr)

    outdir = str(model_dir / 'heatmaps')
    generate_attention_heatmaps(outdir=outdir, dataset=dataset, bags=bags_to_use, attention=y_att, interpolation='bicubic', cmap='inferno')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
