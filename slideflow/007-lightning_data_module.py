import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import slideflow as sf
    import matplotlib.pyplot as plt
    import pandas as pd

    # PyTorch / Lightning imports
    import torch
    from torch.utils.data import DataLoader, Dataset
    import pytorch_lightning as pl

    from utils.sf2torch import DatasetFromSlideFlow
    from utils.mil_lightning import MILDataModule
    return DatasetFromSlideFlow, MILDataModule, sf


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["tumor", "normal"]})
    # dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return P, dataset


@app.cell
def _(dataset):
    train_sf_dataset, valid_sf_dataset = dataset.split(
        model_type='classification', # Categorical labels
        labels='category',            # Label to balance between datasets
        val_strategy='fixed',          # Fixed split strategy
        val_fraction=0.2,             # 20% of data for testing
        splits='/workspace/slideflow_project/splits.json'         # Where to save/load crossfold splits
    )
    return train_sf_dataset, valid_sf_dataset


@app.cell
def _(P):
    test_sf_dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["test"]})
    return (test_sf_dataset,)


@app.cell
def _(
    DatasetFromSlideFlow,
    test_sf_dataset,
    train_sf_dataset,
    valid_sf_dataset,
):
    train_torch_dataset = DatasetFromSlideFlow(train_sf_dataset)
    valid_torch_dataset = DatasetFromSlideFlow(valid_sf_dataset)
    test_torch_dataset = DatasetFromSlideFlow(test_sf_dataset)
    return test_torch_dataset, train_torch_dataset, valid_torch_dataset


@app.cell
def _(
    MILDataModule,
    test_torch_dataset,
    train_torch_dataset,
    valid_torch_dataset,
):
    data_module = MILDataModule(train_torch_dataset, valid_torch_dataset, test_torch_dataset, batch_size=4, num_workers=2)
    return (data_module,)


@app.cell
def _(data_module):
    dl = data_module.train_dataloader()
    X_total, N_sizes, Y_labels = next(iter(dl))
    print("X_total shape:", X_total.shape)  # Expected: (total_tiles, embed_dim)
    print("N_sizes:", N_sizes)              # Expected: [batch_size,]
    print("Y_labels:", Y_labels)            # Expected: (batch_size,)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
