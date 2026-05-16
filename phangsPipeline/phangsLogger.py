import logging
import sys

ALLOWED_LOG_LEVELS = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


def setup_logger(
    level: str = "INFO",
    log_format: str = None,
    logfile: str = None,
) -> logging.Logger:
    """Configure and attach stream/file handlers for this module logger.

    Args:
        level (str): Logging level name used for handler thresholds.
        log_format (str | None): Formatter pattern. If ``None``, a default
            timestamped format is used.
        logfile (str | None): Optional path to a log file. When provided, logs
            are written to both stdout and this file.

    Returns:
        logging.Logger: Configured logger instance for this module.
    """

    # Raise Error if log level is not allowed
    level = level.upper()
    if level not in ALLOWED_LOG_LEVELS:
        raise ValueError(f"level should be one of {ALLOWED_LOG_LEVELS}, not {level}")

    # Set log level
    logger = logging.getLogger()
    logger.setLevel(level)

    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] [%(funcName)s]: %(message)s"
    log_formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up screen handler
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setFormatter(log_formatter)
    screen_handler.set_name("screen_handler")
    logger.addHandler(screen_handler)

    # Set file output and add file handler
    if logfile is not None:
        file_handler = logging.FileHandler(logfile, mode="w+")
        file_handler.setFormatter(log_formatter)
        file_handler.set_name("file_handler")
        logger.addHandler(file_handler)

    return logger
