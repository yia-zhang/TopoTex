# -*- coding: utf-8 -*-
"""Structured logging: one console handler, level from TOPOTEX_LOGLEVEL."""

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "topotex") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("TOPOTEX_LOGLEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
