from .callbacks import CallbackList, Callback, History
from .metrics import Metric, MeanMetric

import torch
from torch import nn

from typing import List, Sequence, Dict
import copy


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("current_train_epoch", torch.zeros([], dtype=torch.int32))
        self.register_buffer("current_train_step", torch.zeros([], dtype=torch.int32))
        self._metrics = None
        self._optimizer = None
        self._loss = None

    def trainable_parameters(self):
        return filter(lambda x: x.requires_grad, self.parameters())

    def non_trainable_parameters(self):
        return filter(lambda x: not (x.requires_grad), self.parameters())

    @property
    def metrics(self) -> Sequence[Metric]:
        return self._metrics if self._metrics else []

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss

    def compile(self, optimizer=None, loss=None, metrics=None):
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = nn.ModuleList(metrics)

    def train_step(self, data):
        raise NotImplementedError

    def test_step(self, data):
        raise NotImplementedError

    def predict_step(self, data):
        raise NotImplementedError

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs,
        validation_loader: torch.utils.data.DataLoader = None,
        callbacks: List[Callback] = None,
        validation_freq=1,
        verbose=2,
        steps_per_epoch=None,
        validation_steps=None,
        device=None,
        dtype=None,
    ):
        callbacks = CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            device=device,
            dtype=dtype,
        )
        self.stop_training = False

        ### on train begin
        logs = {}
        self.on_train_begin(callbacks=callbacks)
        for epoch in range(self.current_train_epoch, epochs):
            ### On epoch begin
            self.on_epoch_begin(epoch=epoch, callbacks=callbacks)

            self.train()
            for batch, train_data in enumerate(train_loader):
                train_data = to(train_data, device=device, dtype=dtype)
                if steps_per_epoch is not None and batch >= steps_per_epoch:
                    break

                ### On batch begin
                self.on_batch_begin(batch=batch, callbacks=callbacks)
                logs = self.train_step(train_data)
                logs = {} if logs is None else logs

                ### On batch end
                self.current_train_step += 1
                self.on_batch_end(batch=batch, callbacks=callbacks, logs=logs)

            self._update_logs_with_metrics(logs)
            logs = copy.copy(logs)

            if validation_loader is not None and self._should_eval(
                epoch=epoch, validation_freq=validation_freq
            ):
                val_logs = self.evaluate(
                    dataloader=validation_loader,
                    callbacks=callbacks,
                    steps=validation_steps,
                    device=device,
                    dtype=dtype,
                )
                val_logs = {f"val_{key}": value for key, value in val_logs.items()}
                logs.update(val_logs)

            ### On epoch end
            self.current_train_epoch += 1
            self.on_epoch_end(epoch=epoch, callbacks=callbacks, logs=logs)
            if self.stop_training:
                break

        ### On train end
        self.on_train_end(logs=logs, callbacks=callbacks)

        return self.history

    def evaluate(
        self,
        dataloader,
        callbacks: List[Callback] = None,
        steps=None,
        verbose=2,
        device=None,
        dtype=None,
    ):
        callbacks = (
            callbacks
            if isinstance(callbacks, CallbackList)
            else CallbackList(
                callbacks=callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                device=device,
                dtype=dtype,
            )
        )

        ### on test begin
        self.on_test_begin(callbacks)

        self.eval()
        for batch, data in enumerate(dataloader):
            data = to(data, device=device, dtype=dtype)
            if steps is not None and batch >= steps:
                break

            ### On batch begin
            self.on_test_batch_begin(batch=batch, callbacks=callbacks)
            with torch.inference_mode():
                logs = self.test_step(data)
                logs = {} if logs is None else logs

            ### On batch end
            self.on_test_batch_end(batch=batch, callbacks=callbacks, logs=logs)

        ### On test end
        self.on_test_end(callbacks=callbacks, logs=logs)

        self._update_logs_with_metrics(logs)

        return logs

    def predict(self, dataloader, callbacks: List[Callback] = None):
        raise NotImplementedError
        callbacks = self._initialize_callbacks(callbacks)

        ### on predict begin
        self.on_predict_begin(callbacks)

        for batch, data in enumerate(dataloader):
            ### On batch begin
            self.on_predict_batch_begin(batch=batch, callbacks=callbacks)
            logs = self.predict_step(data)

            ### On batch end
            self.on_predict_batch_end(batch=batch, callbacks=callbacks, logs=logs)

        ### On predict end
        self.on_predict_end(callbacks=callbacks, logs=logs)

        return logs

    def _should_eval(self, epoch, validation_freq):
        if not validation_freq or validation_freq < 0:
            return False
        return epoch % validation_freq == 0

    ########################################

    def on_test_begin(self, callbacks):
        self._reset_metrics()
        callbacks.on_test_begin()

    def on_test_end(self, callbacks, logs):
        callbacks.on_test_end(logs=logs)

    def on_test_batch_begin(self, batch, callbacks):
        callbacks.on_test_batch_begin(batch=batch)

    def on_test_batch_end(self, batch, callbacks, logs):
        callbacks.on_test_batch_end(batch=batch, logs=logs)

    def on_train_begin(self, callbacks):
        callbacks.on_train_begin()

    def on_train_end(self, callbacks, logs):
        callbacks.on_train_end(logs=logs)

    def on_epoch_begin(self, epoch, callbacks):
        self._reset_metrics()
        callbacks.on_epoch_begin(epoch=epoch)

    def on_epoch_end(self, epoch, callbacks, logs):
        callbacks.on_epoch_end(epoch=epoch, logs=logs)

    def on_batch_begin(self, batch, callbacks):
        callbacks.on_batch_begin(batch=batch)

    def on_batch_end(self, batch, callbacks, logs):
        callbacks.on_batch_end(batch=batch, logs=logs)

    def on_predict_begin(self, callbacks):
        callbacks.on_predict_begin()

    def on_predict_batch_begin(self, batch, callbacks):
        callbacks.on_predict_batch_begin(batch=batch)

    def on_predict_batch_end(self, batch, callbacks, logs):
        callbacks.on_predict_batch_end(batch=batch, logs=logs)

    def on_predict_end(self, callbacks, logs):
        callbacks.on_predict_end(logs=logs)

    ########################################

    def _wrap_around_nn_module(self, x):
        if isinstance(x, Sequence):
            return nn.ModuleList(x)

        if isinstance(x, Dict):
            return nn.ModuleDict(x)

        return x

    def _initialize_callbacks(self, callbacks, in_train=False):
        callbacks = [] if callbacks is None else callbacks

        if in_train:
            callbacks.append(History())

        [callback.set_model(self) for callback in callbacks]
        return callbacks

    def _reset_metrics(self):
        [metric.reset() for metric in self.metrics]

    def _update_logs_with_metrics(self, logs):
        metric_logs = {}
        for metric in self.metrics:
            if metric.name not in logs:
                continue
            result = metric.result()
            if isinstance(result, dict):
                metric_logs.update(result)
            else:
                metric_logs[metric.name] = result

        logs.update(metric_logs)

    def metrics_update_state(self, *args, **kwds):
        with torch.inference_mode():
            [metric.update_state(*args, **kwds) for metric in self.metrics]

    ########################################


class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.loss_tracker = MeanMetric(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker] + list(super().metrics)

    def _graph(self, data):
        x, y = data
        y_pred = self(x)
        loss = self.loss(y_pred=y_pred, y_true=y).mean()

        logs = {}
        with torch.inference_mode():
            self.loss_tracker(loss)
            logs[self.loss_tracker.name] = self.loss_tracker.result()
            for metric in super().metrics:
                metric.update_state(y_pred=y_pred, y_true=y)
                logs[metric.name] = metric.result()

        return loss, logs

    def train_step(self, data):
        loss, logs = self._graph(data)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logs

    def test_step(self, data):
        loss, logs = self._graph(data)
        return logs


def to(data, **kwds):
    if len(kwds) == 0 or all(map(lambda val: val is None, kwds.values())):
        return data

    if isinstance(data, nn.Module) or isinstance(data, torch.Tensor):
        data = data.to(**kwds)
    elif isinstance(data, Sequence):
        data = [to(d, **kwds) for d in data]
    elif isinstance(data, Dict):
        data = {key: to(data[key], **kwds) for key in data}

    return data


def trainable_parameters(parameters):
    return filter(lambda x: x.requires_grad, parameters)


def non_trainable_parameters(parameters):
    return filter(lambda x: not (x.requires_grad), parameters)
