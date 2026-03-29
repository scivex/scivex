"""Logging utilities for pyscivex operations."""
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("pyscivex")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for pyscivex.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(name)s %(levelname)s] %(message)s")
    )
    logger.setLevel(numeric_level)
    if not logger.handlers:
        logger.addHandler(handler)


@contextmanager
def timed(label: str = "operation"):
    """Context manager that logs elapsed time for an operation.

    Usage::

        with timed("training"):
            model.fit(x, y)
    """
    start = time.perf_counter()
    logger.info("%s started", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s finished in %.3fs", label, elapsed)
