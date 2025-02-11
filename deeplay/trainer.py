from __future__ import annotations

from lightning import Trainer as pl_Trainer
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.trainer.connectors.callback_connector import (
    _CallbackConnector,
)

from deeplay.callbacks import LogHistory, RichProgressBar, TQDMProgressBar


class _DeeplayCallbackConnector(_CallbackConnector):
    def _configure_progress_bar(self, enable_progress_bar: bool = True) -> None:

        progress_bars = [
            c for c in self.trainer.callbacks if isinstance(c, ProgressBar)
        ]
        if enable_progress_bar and not progress_bars:
            self.trainer.callbacks.append(TQDMProgressBar())

        # Not great. Should be in a separate configure method. However, this
        # is arguably more stable to api changes in lightning.
        log_histories = [c for c in self.trainer.callbacks if isinstance(c, LogHistory)]
        if not log_histories:
            self.trainer.callbacks.append(LogHistory())

        return super()._configure_progress_bar(enable_progress_bar)


class Trainer(pl_Trainer):

    @property
    def _callback_connector(self):
        return self._callbacks_connector_internal

    @_callback_connector.setter
    def _callback_connector(self, value: _CallbackConnector):
        self._callbacks_connector_internal = _DeeplayCallbackConnector(value.trainer)

    @property
    def history(self) -> LogHistory:
        """Returns the history of the training process."""
        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                return callback
        raise ValueError("History object not found in callbacks")

    def disable_history(self) -> None:
        """Disables the history callback."""
        for callback in self.callbacks:
            if isinstance(callback, LogHistory):
                self.callbacks.remove(callback)
                return
        raise ValueError("History object not found in callbacks")

    def disable_progress_bar(self: Trainer) -> None:
        """Disables the progress bar.

        Raises
        ------
        ValueError
            If no progress bar callback is found.

        """

        for callback in self.callbacks:
            if isinstance(callback, ProgressBar):
                self.callbacks.remove(callback)
                return

        raise ValueError("Progress bar object not found in callbacks")

    def tqdm_progress_bar(self: Trainer, refresh_rate: int = 1) -> None:
        """Enables the TQDM progress bar.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by detault 1. 

        """

        self.disable_progress_bar()

        self.callbacks.append(TQDMProgressBar(refresh_rate=refresh_rate))

    def rich_progress_bar(self, refresh_rate: int = 1, leave: bool = False) -> None:
        """Enables the rich progress bar.

        Parameters
        ----------
        refresh_rate : int, optional
            The refresh rate of the progress bar, by default 1.
        leave : bool, optional
            Whether to leave the progress bar after completion,
            by default False.

        """

        self.disable_progress_bar()

        self.callbacks.append(
            RichProgressBar(refresh_rate=refresh_rate, leave=leave),
        )
