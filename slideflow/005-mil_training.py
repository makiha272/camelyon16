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
    return mo, sf


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["tumor", "normal"]})
    # dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return (dataset,)


@app.cell
def _(dataset):
    train_dataset, valid_dataset = dataset.split(
        model_type='classification', # Categorical labels
        labels='category',            # Label to balance between datasets
        val_strategy='fixed',          # Fixed split strategy
        val_fraction=0.2,             # 20% of data for testing
        splits='/workspace/slideflow_project/splits.json'         # Where to save/load crossfold splits
    )
    return train_dataset, valid_dataset


@app.cell
def _():
    from slideflow.mil import mil_config
    config = mil_config(
        model='attention_mil',
        lr=1e-4,
        batch_size=8,
        epochs=20,
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
        outdir='/workspace/slideflow_project/models',
        attention_heatmap=True
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## TensorBoard併用
    - 用意されていないので実装する必要あり
    """)
    return


@app.cell
def _():
    # from fastai.callback.tensorboard import TensorBoardCallback
    # from slideflow.mil.train import build_fastai_learner, _fastai  # _fastai.train handles fitting

    # # Build learner (returns a fastai Learner)
    # learner = build_fastai_learner(
    #     config,
    #     train_dataset=train_dataset,
    #     val_dataset=valid_dataset,
    #     outcomes='category',
    #     bags='/workspace/slideflow_project/bags',
    #     outdir='/workspace/slideflow_project/models'  # where outputs go
    # )

    # # Debug: inspect dls and batch format so we can verify `xb` contains `bags, lens`.
    # print('dls.n_inp:', getattr(learner.dls, 'n_inp', None))
    # print('Validation dataset type:', type(learner.dls.valid.dataset))
    # try:
    #     b = learner.dls.valid.one_batch()
    #     print('valid batch length:', len(b))
    #     print('valid batch types:', [type(x) for x in b])
    # except Exception as e:
    #     print('Could not inspect one_batch():', e)

    # # Create a TensorBoard callback and train
    # # - trace_model=False avoids running a dummy forward to generate a graph (prevents lens-related errors)
    # # - log_preds=False disables call to `show_results` on the dataset (our MapDataset lacks `show_results`)
    # #   which prevents the AttributeError you saw: "'MapDataset' object has no attribute 'show_results'"

    # tb_cb = TensorBoardCallback(
    #     log_dir='/workspace/slideflow_project/tensorboard_logs',
    #     trace_model=False,
    #     log_preds=False
    # )
    # _fastai.train(learner, config, callbacks=[tb_cb])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
