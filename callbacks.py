from torch.utils.tensorboard import SummaryWriter
import torch
from pathlib import Path
import time
import sys
import numpy as np

from typing import Sequence


class Callback:
    def __init__(self):
        self.model = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """


class CallbackList:
    def __init__(
        self,
        callbacks: Sequence[Callback] = None,
        add_history=False,
        add_progbar=False,
        model=None,
        **params,
    ):
        self.model = model
        self.params = params
        self.callbacks = list(callbacks) if callbacks is not None else []
        if add_history:
            self.callbacks.append(History())

        if add_progbar:
            self.callbacks.append(Progbar())

        for callback in self.callbacks:
            callback.set_model(model)
            callback.set_params(params)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch=batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch=epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch=epoch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch=batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_test_batch_end(batch=batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch=batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch=batch, logs=logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs=logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs=logs)

    def on_predict_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_begin(logs=logs)

    def on_predict_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_end(logs=logs)


class Progbar(Callback):
    def __init__(self, stdout=None):
        super().__init__()
        self.stdout = sys.stdout if stdout is None else stdout

    def on_epoch_begin(self, epoch, logs=None):
        digits = int(np.log10(self.params["epochs"])) + 1
        self.epoch_start_time = time.perf_counter()
        print(
            f"Epoch {epoch:{digits}.0f}/{self.params['epochs']:{digits}.0f}",
            file=self.stdout,
        )

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.perf_counter()
        elapsed_time = epoch_end_time - self.epoch_start_time
        elapsed_time = self.compute_time(elapsed_time)
        print(f"{elapsed_time}", end="", file=self.stdout)
        for key, val in logs.items():
            print(" - ", end="", file=self.stdout)
            print(f"{key}: {val:.4f}", end="", file=self.stdout)
        print(file=self.stdout)

    def compute_time(self, elapsed_time):
        if elapsed_time < 0:
            elapsed_time = elapsed_time * 1000
            return f"{int(elapsed_time)} ms"
        else:
            return f"{elapsed_time:.4f} s"


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.

    Example:

    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, verbose=1)
    >>> print(history.params)
    {'verbose': 1, 'epochs': 10, 'steps': 1}
    >>> # check the keys of history object
    >>> print(history.history.keys())
    dict_keys(['loss'])

    """

    def __init__(self):
        super().__init__()
        self.history = {}

    def set_params(self, params):
        super().set_params(params)
        self.model.history = self

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self


class TensorBoardCallback(Callback):
    def __init__(self, logdir) -> None:
        super().__init__()
        self.logdir = Path(logdir)
        self.is_in_train = False
        self.train_writer = None
        self.val_writer = None
        self.metric_name = set()

    def set_model(self, model):
        super().set_model(model)
        self.metric_name = set([metric.name for metric in self.model.metrics])

    def on_train_begin(self, logs=None):
        self.train_writer = SummaryWriter(str(self.logdir / "train"))

    def on_train_end(self, logs=None):
        self.train_writer.close()
        if self.val_writer is not None:
            self.val_writer.close()
            self.val_writer = None

    def _push(self, writer, logs, global_step):
        if len(logs) == 0:
            return

        for key, val in logs.items():
            writer.add_scalar(tag=key, scalar_value=val, global_step=global_step)

    def on_epoch_end(self, epoch, logs=None):
        train_logs = {
            key: val for key, val in logs.items() if not key.startswith("val_")
        }
        train_logs["steps"] = self.model.current_train_step
        val_logs = {key[4:]: val for key, val in logs.items() if key.startswith("val_")}

        self._push(writer=self.train_writer, logs=train_logs, global_step=epoch)

        if len(val_logs) == 0:
            return

        self.val_writer = (
            self.val_writer
            if self.val_writer is not None
            else SummaryWriter(str(self.logdir / "val"))
        )
        self._push(writer=self.val_writer, logs=val_logs, global_step=epoch)

    def on_batch_end(self, batch, logs=None):
        global_step = self.model.current_train_step

        logs = {
            key: logs[key] for key in logs if logs.keys() if key not in self.metric_name
        }
        self._push(writer=self.train_writer, logs=logs, global_step=global_step)


class CheckPointCallback(Callback):
    def __init__(
        self,
        save_path,
        prefix="",
        map_location=None,
        **modules,
    ) -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self.modules = modules
        self.prefix = prefix
        self.map_location = map_location
        self.epoch = 0
        modules["ckpt"] = self

    def restore(self):
        try:
            checkpoint = torch.load(
                self.save_path / f"{self.prefix}ckpt.pt", map_location=self.map_location
            )
        except FileNotFoundError as err:
            print("Initializing from scratch.")
            return

        for name, module in self.modules.items():
            if hasattr(module, "load_state_dict"):
                module.load_state_dict(checkpoint[name])
            else:
                self.modules[name] = checkpoint[name]

        load_path = self.save_path.joinpath(f"{self.prefix}ckpt.pt")
        print(f"Restored from {str(load_path)}")

    def save(self):
        to_be_saved = {
            name: module.state_dict() for name, module in self.modules.items()
        }
        torch.save(
            to_be_saved,
            self.save_path / f"{self.prefix}ckpt.pt",
        )

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        self.save()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {f"{prefix}epoch": self.epoch}

    def load_state_dict(self, state_dict, strict=True):
        self.epoch = state_dict["epoch"]


class BestCheckPointCallback(CheckPointCallback):
    def __init__(
        self,
        monitor,
        save_path,
        prefix="",
        min_delta=0.0001,
        min_is_better=True,
        map_location=None,
        **modules,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.min_is_better = min_is_better
        self.best_score = np.inf

        super().__init__(
            save_path=save_path,
            map_location=map_location,
            prefix=prefix,
            **modules,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict[f"{prefix}best_score"] = self.best_score
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict=state_dict, strict=strict)
        self.best_score = state_dict["best_score"]

    def on_epoch_end(self, epoch, logs=None):
        var = logs[self.monitor]

        if not self.min_is_better:
            var = -var

        if var + self.min_delta < self.best_score:
            self.best_score = var
            super().on_epoch_end(epoch=epoch, logs=logs)
