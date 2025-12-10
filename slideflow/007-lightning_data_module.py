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
    return DataLoader, Dataset, pl, sf, torch


@app.cell
def _(sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["tumor", "normal"]})
    # dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})
    return P, dataset


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
def _(P):
    test_dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["test"]})
    return (test_dataset,)


@app.cell
def _(Dataset, sf, torch, train_dataset):
    class DatasetFromSlideFlow(Dataset):
        def __init__(self, sf_dataset: sf.Dataset):
            self.sf_dataset = sf_dataset
            self.slides = sf_dataset.slides()
            self.annotations = self.sf_dataset.annotations
            self.bag_path = "/workspace/slideflow_project/bags/"
            #①スライドとラベル(encoded)を対応させた辞書 ②ラベルとそのencodingを対応させた辞書 を作成
            self.slide_labels, self.label_map = train_dataset.labels(headers="category")

        def __len__(self):
            return len(self.slides)

        def __getitem__(self, idx):
            slide_name = self.slides[idx]
            slide_bag = torch.load(self.bag_path + slide_name + ".pt")
            label_encoded = self.slide_labels[slide_name]
            return slide_bag, label_encoded

    return (DatasetFromSlideFlow,)


@app.cell
def _(DataLoader, DatasetFromSlideFlow, pl):
    class MILDataModule(pl.LightningDataModule):
        def __init__(self, train_dataset, valid_dataset, test_dataset, batch_size=1, num_workers=4):
            super().__init__()
            self.train_dataset = DatasetFromSlideFlow(train_dataset)
            self.valid_dataset = DatasetFromSlideFlow(valid_dataset)
            self.test_dataset = DatasetFromSlideFlow(test_dataset)
            self.batch_size = batch_size
            self.num_workers = num_workers

        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        def val_dataloader(self):
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    return (MILDataModule,)


@app.cell
def _(MILDataModule, test_dataset, train_dataset, valid_dataset):
    data_module = MILDataModule(train_dataset, valid_dataset, test_dataset, batch_size=1, num_workers=2)
    return (data_module,)


@app.cell
def _(data_module):
    dl = data_module.train_dataloader()
    for bags, labels in dl:
        print("Bags shape:", bags.shape)  # Expected: (batch_size, num_tiles, channels, height, width)
        print("Labels shape:", labels.shape)  # Expected: (batch_size,)
        break
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
