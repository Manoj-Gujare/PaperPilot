"""Tests for logger configuration."""

from __future__ import annotations

import logging

from paperpilot.core.logging import configure_logging, get_logger


def test_configure_logging_attaches_a_single_handler():
    configure_logging("DEBUG", force=True)
    configure_logging("DEBUG")
    logger = logging.getLogger("paperpilot")
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_logger_names_are_namespaced():
    assert get_logger("paperpilot.ingestion").name == "paperpilot.ingestion"
    assert get_logger("ingestion").name == "paperpilot.ingestion"


def test_records_do_not_propagate_to_root():
    configure_logging("INFO", force=True)
    assert logging.getLogger("paperpilot").propagate is False
