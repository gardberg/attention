import logging

import os

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    print("Invalid log level provided. Using INFO as the default.")
    log_level = "INFO"

# Custom logger
log_format = "\n%(levelname)s: %(message)s"
formatter = logging.Formatter(log_format)

logger = logging.getLogger("custom_logger")
logger.setLevel(log_level)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.propagate = False
logger.addHandler(handler)
