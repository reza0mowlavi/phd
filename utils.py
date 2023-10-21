import torch
import numpy as np


class LinearScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate,
        maximal_learning_rate,
        left_steps,
        right_steps,
        decay_steps,
        minimal_learning_rate,
    ) -> None:
        self._initial_learning_rate = initial_learning_rate
        self._maximal_learning_rate = maximal_learning_rate
        self._left_steps = left_steps
        self._right_steps = right_steps
        self._decay_steps = decay_steps
        self._lowest_learning_rate = minimal_learning_rate

        left = np.linspace(initial_learning_rate, maximal_learning_rate, left_steps)
        right = np.linspace(maximal_learning_rate, initial_learning_rate, right_steps)
        decay = np.linspace(initial_learning_rate, minimal_learning_rate, decay_steps)
        self._lr = np.concatenate((left, right, decay))
        self.cycle_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> float:
        lr = (
            self._lr[self.cycle_step]
            if self.cycle_step < len(self._lr)
            else self._lr[-1]
        )
        self.cycle_step = (
            self.cycle_step + 1 if self.cycle_step < len(self._lr) else self.cycle_step
        )
        return [lr]
