import logging

LOG_LEVEL = 25
# LOG_LEVEL = logging.DEBUG


def get_logger():
    # Creates a local logger for the file it is called in with __name__
    # as the name of the logger.
    logger = logging.getLogger(__name__)

    # set format
    log_format = "\n\x1b[35mDEBUG\x1b[0m: %(message)s"
    formatter = logging.Formatter(log_format)
    # create a handler
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.log(LOG_LEVEL, "Logging working")
