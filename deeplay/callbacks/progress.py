from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar as RPB,
    RichProgressBarTheme as RPBT,
)

from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar as TQDM


# from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar as TQDM


class TQDMProgressBar(TQDM):
    def __init__(
        self,
        refresh_rate: int = 1,
    ):
        super().__init__(
            refresh_rate=refresh_rate,
        )


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
