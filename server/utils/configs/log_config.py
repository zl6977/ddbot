# log_config.py
import logging
import os
from logging.handlers import RotatingFileHandler

from . import globals_config as glb


def configure_logger(log_file_path: str, mode: str = "w", reset: bool = False, logging_level=logging.INFO):
    """
    Set up the logger with a file and stream handler.

    :param log_file_path: Path to the log file.
    :param mode: File mode, default is 'w' (overwrite).
    :param reset: If True, removes existing handlers before configuring.
    :param logging_level: The lowest level to be logged.
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if reset:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_file_path, mode=mode, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def change_logfile_path(log_file_path: str, mode: str = "w"):
    """
    This function is to be deprecated.
    change log file
    :param log_file_path: log file path
    """

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[RotatingFileHandler(log_file_path, mode=mode, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"), logging.StreamHandler()],
    )
    # logging.addHandler(logging.FileHandler(log_file_path))
    # logging.addHandler(logging.StreamHandler())


configure_logger(glb.default_log_path, mode="w")
