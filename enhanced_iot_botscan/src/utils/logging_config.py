"""
Centralized logging configuration for Enhanced IoT BotScan.

Configures both console and file logging. Log files are stored in the
`logs/` directory and rotated automatically once they exceed 10 MB.

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()   # call once at app startup
"""

import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'iot_botscan.log')

# Max 10 MB per file, keep 3 backups (iot_botscan.log, .log.1, .log.2, .log.3)
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 3


def setup_logging(level: str = 'INFO',
                  log_file: str = None,
                  fmt: str = None) -> None:
    """
    Configure root logger with console + rotating file handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Override default log file path
        fmt: Override default log format string
    """
    log_file = log_file or LOG_FILE
    fmt = fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure log directory exists
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid adding duplicate handlers on re-import / re-run
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    formatter = logging.Formatter(fmt)

    # ---- File handler (rotating) ----
    file_handler = RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ---- Console handler (already present via basicConfig in some cases) ----
    # Only add if there isn't already a StreamHandler
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    logging.info(f"Logging initialized → {os.path.abspath(log_file)}")
