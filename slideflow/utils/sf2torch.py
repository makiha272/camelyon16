# PyTorch / Lightning imports
import torch
from torch.utils.data import Dataset

import slideflow as sf


class DatasetFromSlideFlow(Dataset):
    def __init__(self, sf_dataset: sf.Dataset):
        self.sf_dataset = sf_dataset
        self.slides = sf_dataset.slides()
        self.annotations = self.sf_dataset.annotations
        self.bag_path = "/workspace/slideflow_project/bags/"
        # ①スライドとラベル(encoded)を対応させた辞書 ②ラベルとそのencodingを対応させた辞書 を作成
        self.slide_labels, self.label_map = sf_dataset.labels(headers="category")

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide_name = self.slides[idx]
        slide_bag = torch.load(self.bag_path + slide_name + ".pt")
        # Ensure features are tensors (torch.cat in collate expects tensors)
        assert torch.is_tensor(slide_bag)
        label_encoded = self.slide_labels[slide_name]
        if not torch.is_tensor(label_encoded):
            label_encoded = torch.tensor(label_encoded, dtype=torch.long)
        return slide_bag, label_encoded
