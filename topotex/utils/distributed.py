# -*- coding: utf-8 -*-
"""torchrun environment helpers."""

import os


def ddp_env():
    """(rank, world_size, local_rank); (0, 1, 0) when not under torchrun."""
    if "RANK" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ["LOCAL_RANK"]),
        )
    return 0, 1, 0
