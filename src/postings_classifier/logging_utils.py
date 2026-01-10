"""Logging utilities for the postings classifier project."""
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(log_dir: str = "logs", level: str = "INFO", rotation: str = "10 MB") -> None:
    """Configure loguru to log to stdout and a rotating file.

    Args:
        log_dir: Directory to write log files.
        level: Minimum log level to emit.
        rotation: File size or time-based rotation setting.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    logger.add(log_path / "app.log", level=level, rotation=rotation, enqueue=True)


def log_config(cfg: Any) -> None:
    """Log a summarized view of the Hydra config."""
    try:
        import omegaconf

        cfg_str = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
        logger.debug("Configuration:\n{}", cfg_str)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to log config: {}", exc)
