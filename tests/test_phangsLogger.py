import logging
import os

import pytest
from pytest import LogCaptureFixture

from phangsPipeline import setup_logger


class TestPhangsLogger:
    """Suite of tests for the phangsLogger"""

    log_levels = [
        pytest.param(
            "SHOULD NOT WORK",
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="Not a valid log level",
            ),
        ),
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]

    @pytest.mark.parametrize("level", log_levels)
    def test_log_level(
        self,
        level: str,
    ):
        """Test log level is as expected

        Args:
            level (str): Log level
        """

        logger = setup_logger(level=level)

        assert logger.getEffectiveLevel() == getattr(logging, level)

    def test_log_format(
            self,
    ):
        """Test log format is passed as expected"""

        log_format = "[%(levelname)s]: %(message)s"

        logger = setup_logger(log_format=log_format)

        found_format = None
        for handler in logger.handlers:
            if handler.name == "screen_handler":
                found_format = handler.formatter._fmt

        assert log_format == found_format

    def test_logfile_exists(self):
        """Test passing a logfile name actually creates a file"""

        logfile = "test_logfile.log"

        # Remove existing logfile if it already exists
        if os.path.exists(logfile):
            os.remove(logfile)

        setup_logger(logfile=logfile)

        # Check logfile exists, then remove
        success = os.path.exists(logfile)
        if os.path.exists(logfile):
            os.remove(logfile)

        assert success

    def test_log_output(
        self,
        caplog: LogCaptureFixture,
    ):
        """Test logging output

        Args:
            caplog (LogCaptureFixture): LogCaptureFixture
        """

        caplog.set_level(logging.INFO)

        log_str = "Test log output"

        logger = setup_logger(level="INFO")
        logger.info(log_str)

        assert log_str in caplog.text
