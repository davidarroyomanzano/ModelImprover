import logging
from unittest.mock import patch

from ds_project import logger


def test_get_logger_returns_named_logger():
    log = logger.get_logger("x.y")
    assert isinstance(log, logging.Logger)
    assert log.name == "x.y"


def test_logging_configured_only_once(monkeypatch):
    monkeypatch.setattr(logger, "_configured", False)
    with patch("logging.config.dictConfig") as m:
        logger.get_logger("a")
        logger.get_logger("b")
    m.assert_called_once()
