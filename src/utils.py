import logging
import os
import jax
from typing import Union
from attention import NamedTupleSubclass

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

        
def state_to_str(state: Union[NamedTupleSubclass, jax.Array, bool], indent=0):
    # state is a NamedTuple which contains several other NamedTuples, jax.Arrays, or bools

    if state is None: return "None"

    if isinstance(state, jax.Array): return f"{state.shape}"

    if isinstance(state, bool): return state

    result = [f"{state.__class__.__name__}:"] if state else []
    for name, value in state._asdict().items():
        field_str = f"\t{name}: " if indent > 0 else f"{name}: "
        result.append(f"\t{field_str}{state_to_str(value, indent + 1)}")

    return "\n".join(result)


if __name__ == "__main__":
    logger = get_logger()
    logger.log(LOG_LEVEL, "Logging working")
