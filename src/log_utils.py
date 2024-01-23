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
print("Using log level:", curr_log_level)

logger.info("logger.info at level: %s", curr_log_level)
logger.debug("logger.debug at level: %s", curr_log_level)
logger.warning("logger.warning at level: %s", curr_log_level)
