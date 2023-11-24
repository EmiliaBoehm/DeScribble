"""
Logger utilities

Calling logger.set_logger() creates a global variable 'log' with a logger object.
"""

import logging

# -----------------------------------------------------------
# Logging


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a logger object."""
    log = logging.getLogger(name)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(module)s: %(funcName)s() - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(f"{name}.log", mode='w')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    #log.addHandler(fh)
    log.addHandler(ch)
    log.setLevel(logging.DEBUG)  # all log messages to the handler
    fh.setLevel(level)
    ch.setLevel(level)
    return log


def set_logger(level=logging.INFO) -> logging.Logger:
    """Set a global logging variable if it is not already set."""
    if 'log' in globals():
        for handler in globals()['log'].handlers:
            handler.close()
            globals()['log'].removeHandler(handler)
        del globals()['log']
        return get_logger('Logger', level)
    else:
        return get_logger('Logger', level)
