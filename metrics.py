import torch
from torch import nn


class Metric(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwds):
        self.update_state(*args, **kwds)
        return self.result()

    def reset(self):
        raise NotImplementedError

    def update_state(self, *args, **kwds):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class MeanMetric(Metric):
    def __init__(self, name):
        super().__init__(name=name)
        self.total = 0
        self.count = 0

    @torch.inference_mode()
    def update_state(self, y):
        self.count += y.size(0) if y.dim() > 0 else 1
        self.total += y.sum().cpu().item()

    def result(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def reset(self):
        self.total = 0
        self.count = 0


class BinaryAccuracyMetric(Metric):
    def __init__(self, threshold, name):
        super().__init__(name=name)
        self.threshold = threshold
        self.count = 0
        self.correct = 0

    @torch.inference_mode()
    def update_state(self, y_true, y_pred):
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        y_pred = y_pred >= self.threshold

        self.count += y_true.size(0)
        self.correct += (y_pred == y_true).sum().cpu().item()

    def result(self):
        if self.count == 0:
            return 0
        return self.correct / self.count

    def reset(self):
        self.count = 0
        self.correct = 0
