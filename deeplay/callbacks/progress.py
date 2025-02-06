from weakref import ref
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar as RPB,
    RichProgressBarTheme as RPBT,
)

from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar as TQDM

from lightning.pytorch.utilities.rank_zero import rank_zero_debug
import os
import warnings

# from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar as TQDM


class TQDMProgressBar(TQDM):
    def __init__(
        self,
        refresh_rate: int = 1,
    ):
        super().__init__(
            refresh_rate=refresh_rate,
        )

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if "COLAB_JUPYTER_IP" in os.environ and refresh_rate == 1:
            rank_zero_debug(
                "Small refresh rates can crash on Colab or Kaggle. Setting refresh_rate to 10.\n"
                "To manually set the refresh rate, call `trainer.tqdm_progress_bar(refresh_rate=10)`."
            )
            refresh_rate = 10

        return TQDM._resolve_refresh_rate(refresh_rate)


class RichProgressBar(RPB):

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RPBT = RPBT(metrics_format=".3g"),
        console_kwargs=None,
    ):
        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme,
            console_kwargs=console_kwargs,
        )

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if "COLAB_JUPYTER_IP" in os.environ and refresh_rate == 1:
            rank_zero_debug(
                "Small refresh rates can crash on Colab or Kaggle. Setting refresh_rate to 10.\n"
                "To manually set the refresh rate, call `trainer.rich_progress_bar(refresh_rate=10)`."
            )
            refresh_rate = 10

        return TQDM._resolve_refresh_rate(refresh_rate)
