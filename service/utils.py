import importlib
import logging
import os
import socket
import sys
import types


# Get hostname
_HOSTNAME_FULL = socket.gethostname()
_HOSTNAME_SPLIT_MARKS = (".", "_", "-")
_HOSTNAME = _HOSTNAME_FULL
for mark in _HOSTNAME_SPLIT_MARKS:
    if mark in _HOSTNAME_FULL:
        _HOSTNAME = _HOSTNAME_FULL.rsplit(mark)[-1]


# Custom LogRecord factory, add hostname field
def custom_log_record_factory(*args, **kwargs):
    """Create a custom LogRecord with hostname field.

    Args:
        *args: Variable length argument list for LogRecord.
        **kwargs: Arbitrary keyword arguments for LogRecord.

    Returns:
        logging.LogRecord:
            LogRecord instance with hostname field added.
    """
    record = logging.LogRecord(*args, **kwargs)
    record.hostname = _HOSTNAME
    return record


logging.setLogRecordFactory(custom_log_record_factory)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(thread)d - %(message)s"  # noqa: E501
)


def setup_logger(
    logger_name: str = "root",
    file_level: int = logging.INFO,
    console_level: int = logging.INFO,
    logger_level: int | None = None,
    logger_path: str | None = None,
    logger_format: str | None = None,
) -> logging.Logger:
    """Setup and configure logger.

    Create a logger with specified configuration, supporting simultaneous output
    to console, file, and AWS CloudWatch, with separate log levels for each
    output stream.

    Args:
        logger_name (str, optional):
            Logger name. Defaults to 'root'.
        file_level (int, optional):
            Log level for file output stream. Defaults to logging.INFO.
        console_level (int, optional):
            Log level for console output stream. Defaults to logging.INFO.
        logger_level (int | None, optional):
            Global logger level, deprecated. Use file_level/console_level
            instead. Defaults to None.
        logger_path (str | None, optional):
            Log file path. If None, no file output. Defaults to None.
        logger_format (str | None, optional):
            Log format string. If None, uses default format. Defaults to None.

    Returns:
        logging.Logger:
            Configured logger instance.

    Note:
        logger_level parameter is deprecated. Use file_level and console_level
        to set log levels for different output streams separately.
        Program will automatically gracefully shutdown all CloudWatch log
        handlers on exit to avoid warnings.
    """
    logger = logging.getLogger(logger_name)
    level_candidates = [file_level, console_level]
    if logger_level is not None:
        level_candidates = [
            logger_level,
        ]
        console_level = logger_level
        file_level = logger_level
    min_level = min(level_candidates)
    logger.setLevel(level=min_level)
    # prevent logging twice in stdout
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    handlers = [stream_handler]
    if logger_path is not None:
        handler = logging.FileHandler(logger_path, encoding="utf-8")
        handler.setLevel(file_level)
        handlers.append(handler)
    if logger_format is not None:
        formatter = logging.Formatter(logger_format)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(thread)d - %(message)s"  # noqa: E501
        )
    # assure handlers are not double
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if logger_level is not None:
        logger.warning(
            "UserWarning: logger_level is now deprecated, " +
            "please specify file_level/console_level instead."
        )
    return logger


def get_logger(logger: None | str | logging.Logger = None) -> logging.Logger:
    """Get logger instance.

    Get corresponding logger instance based on input parameter,
    supporting multiple input types.

    Args:
        logger (None | str | logging.Logger, optional):
            Logger identifier. None means get root logger, string means
            logger name. Defaults to None.

    Returns:
        logging.Logger:
            Corresponding logger instance.
    """
    if logger is None or isinstance(logger, str):
        ret_logger = logging.getLogger(logger)
    else:
        ret_logger = logger
    return ret_logger


def file2dict(file_path: str) -> dict:
    """Convert a python file to a dict.

    Args:
        file_path (str): The path of the file.
    Returns:
        dict: The dict converted from the file.
    """
    file_name = os.path.basename(file_path)
    file_name = file_name.split(".")[0]
    file_dir = os.path.dirname(file_path)
    sys.path.insert(0, file_dir)
    file_module = importlib.import_module(file_name)
    sys.path.pop(0)
    file_dict = {
        name: value
        for name, value in file_module.__dict__.items()
        if not name.startswith("__")
        and not isinstance(value, types.ModuleType)
        and not isinstance(value, types.FunctionType)
    }
    return file_dict