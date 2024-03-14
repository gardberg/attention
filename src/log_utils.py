import logging
import logging.config
import os

# Define logger
DEFAULT_LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"

LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

try:
    LOG_LEVEL = getattr(logging, LOG_LEVEL_STR.upper())
except AttributeError:
    print(f"Invalid log level: {LOG_LEVEL_STR}. Defaulting to {DEFAULT_LOG_LEVEL}.")
    LOG_LEVEL = getattr(logging, DEFAULT_LOG_LEVEL.upper())

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
)

# Disable existing loggers
# Might not catch earlier printed logs by other imported modules.
# Is there a way to make sure this is run after all imports?
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
})

logger = logging.getLogger("attention")
logger.setLevel(LOG_LEVEL)

curr_log_level = logging.getLevelName(logger.getEffectiveLevel())

if curr_log_level == "DEBUG":
    logger.debug("Log level set to DEBUG")
elif curr_log_level == "INFO":
    logger.info("Log level set to INFO")
elif curr_log_level == "WARNING":
    logger.warning("Log level set to WARNING")
