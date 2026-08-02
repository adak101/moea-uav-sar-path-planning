"""
Konfiguracja logowania dla projektu.
"""
import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """
    Skonfiguruj logging.

    Args:
        level: Poziom logowania (DEBUG, INFO, WARNING, ...)
        log_file: Opcjonalna ścieżka do pliku logów
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )