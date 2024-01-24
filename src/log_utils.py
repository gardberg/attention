import logging
import os

# Define logger
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"

LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR.upper())

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
)

logger = logging.getLogger("attention")
logger.setLevel(LOG_LEVEL)

curr_log_level = logging.getLevelName(logger.getEffectiveLevel())

if curr_log_level == "DEBUG":
    logger.debug("Log level set to DEBUG")
elif curr_log_level == "INFO":
    logger.info("Log level set to INFO")
elif curr_log_level == "WARNING":
    logger.warning("Log level set to WARNING")
