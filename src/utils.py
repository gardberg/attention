import logging
import os
import jax

os.environ["TIKTOKEN_CACHE_DIR"] = "../.cache"
import tiktoken

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

    
class Tokenizer:
    # Wrapper class for tiktoken tokenizer
    
    def __init__(self, name='cl100k_base'):
        try:
            self.encoding = tiktoken.get_encoding(name)
        except:
            raise Exception(f"Could not download or find tiktoken tokenizer: '{name}'")

    def encode(self, text: str) -> jax.Array:
        enc_list = self.encoding.encode(text)
        return jax.numpy.array(enc_list)

    # TODO: How do we handle batched array?
    def decode(self, enc: jax.Array) -> str:
        return self.encoding.decode(enc.tolist())


if __name__ == "__main__":
    logger = get_logger()
    logger.log(LOG_LEVEL, "Logging working")
