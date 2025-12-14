# PyTorch / Lightning imports
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MILDataModule(pl.LightningDataModule):
    """ "A PyTorch Lightning DataModule for MIL datasets."""

    def __init__(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        test_dataset: Dataset,
        batch_size=1,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._mil_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._mil_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._mil_collate_fn,
        )

    def _mil_collate_fn(self, batch):
        print("here")
        # batch は [(features_1, label_1), (features_2, label_2), ...] のリスト
        all_instance_features = []  # すべてのインスタンス特徴を格納
        bag_labels = []  # バッグラベルを格納
        bag_sizes = []  # 各バッグのインスタンス数 N_i を格納

        for features, label in batch:
            all_instance_features.append(features)
            bag_labels.append(label)
            bag_sizes.append(features.shape[0])

        # 1. すべてのインスタンス特徴を連結 (N_total, embed_dim)
        X_batch = torch.cat(all_instance_features, dim=0)

        # 2. バッグラベルをスタック (B, 1)
        Y_batch = torch.stack(bag_labels).view(-1, 1)

        # 3. バッグサイズをテンソルに (B,)
        N_batch = torch.LongTensor(bag_sizes)

        # モデルには (連結された特徴, バッグサイズ, バッグラベル) を渡す
        return X_batch, N_batch, Y_batch
