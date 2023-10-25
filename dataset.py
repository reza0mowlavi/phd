from pathlib import Path
from PIL import Image

import numpy as np

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
        self.images = {path: Image.open(Path(path)) for path in self.paths}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.images[path] if self.cached else Image.open(Path(path))
        img = self.transform(img) if self.transform is not None else img
        label = self.labels[index]
        return img, label


class OversampledDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths, labels, cached=True, transform=None, max_steps=None):
        self.paths = paths
        self.labels = labels
        self.cached = cached
        self.transform = transform
        self.max_steps = max_steps
        self.iterable_datasets = [
            IterableDataset(
                paths=paths[labels == unique_label],
                labels=labels[labels == unique_label],
                cached=cached,
                transform=transform,
            )
            for unique_label in np.unique(labels)
        ]

    def __iter__(self):
        step = 1
        for elements in zip(*self.iterable_datasets):
            yield elements
            if self.max_steps is not None and self.max_steps < step:
                break
            else:
                step += 1


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths, labels, cached=True, transform=None):
        self.paths = paths
        self.labels = labels
        self.cached = cached
        self.transform = transform
        if self.cached:
            self.cache()

    def cache(self):
        self.images = {path: Image.open(Path(path)) for path in self.paths}

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        indices = np.arange(len(self))
        while True:
            np.random.shuffle(indices)
            for index in indices:
                path = self.paths[index]
                img = self.images[path] if self.cached else Image.open(Path(path))
                img = self.transform(img) if self.transform is not None else img
                label = self.labels[index]
                yield img, label
