from pathlib import Path
from PIL import Image

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, cached=True, transform=None):
        self.paths = paths
        self.labels = labels
        self.cached = cached
        self.transform = transform
        if self.cached:
            self.cache()

    def cache(self):
        self.images = [Image.open(Path(path)) for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths
        img = self.images[index] if self.cached else Image.open(Path(path))
        img = self.transform(img) if self.transform is not None else img
        label = self.labels[index]
        return img, label
