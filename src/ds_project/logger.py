"""Logger centralizado."""

import logging
import logging.config
from pathlib import Path

import yaml

_configured = False

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado desde configs/logging.yaml."""
    global _configured

    if not _configured:
        with open(CONFIG_DIR / "logging.yaml", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
        _configured = True

    return logging.getLogger(name)
