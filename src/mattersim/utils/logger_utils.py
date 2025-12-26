# -*- coding: utf-8 -*-
import os
import sys

from loguru import logger

handlers = {}


def get_logger():
    if not handlers:
        logger.remove()
        enqueue = os.environ.get("MATTERSIM_LOGURU_ENQUEUE", "1") != "0"
        handlers["console"] = logger.add(
            sys.stdout,
            colorize=True,
            filter=log_filter,
            enqueue=enqueue,
        )

    return logger


def log_filter(record):
    if record["level"].name != "INFO":
        return True

    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False
