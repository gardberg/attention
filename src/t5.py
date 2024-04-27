
# T5 translation model based on google t5-small for translation
# takes in input ids in the form of tokens, and returns predicted tokens (for translation)
from attention import Embedding, RMSNorm
from typing import Callable
from states import T5DenseState, T5FeedForwardState
from jax import Array, random
from attention import Linear
from act import relu, dropout

# https://github.com/huggingface/transformers/blob/e4ea19b958c89d61e42461fac6ac8441787121f8/src/transformers/models/t5/modeling_t5.py#L646
class T5Encoder:
    def __init__(self):
        pass

class T5Model:
    def __init__(self, vocab_size: int, emb_size: int):
        self.shared_embedding = Embedding(vocab_size, emb_size) 
        self.encoder = None
        self.decoder = None


class T5Dense:
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1, act: Callable=relu):
        self.wi = Linear(n_in, d_ff, bias=False)
        self.wo = Linear(d_ff, n_in, bias=False)
        self.dropout = dropout
        self.act = act

    def __call__(self, state: T5DenseState, x: Array, rng: Array, training=True) -> Array:
        x = self.wi(state.wi, x)
        x = self.act(x)
        x = dropout(x, self.dropout, rng, training)
        x = self.wo(state.wo, x)
        return x

    def init_state(self, rng: Array) -> T5DenseState:
        rng1, rng2 = random.split(rng, 2)
        return T5DenseState(
            wi=self.wi.init_state(rng1),
            wo=self.wo.init_state(rng2),
        )


class T5FeedForward:
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1):
        self.dense = T5Dense(n_in, d_ff, dropout)
        # T5LayerFF uses custom LayerNorm which is equivalent to RMSNorm
        self.norm = RMSNorm(n_in, eps=1e-6)
        self.dropout = dropout

    def __call__(self, state: T5FeedForwardState, x: Array, rng: Array, training=True) -> Array:
        rng1, rng2 = random.split(rng, 2)

        z = self.norm(state.norm, x)
        z = self.dense(state.dense, z, rng1, training)
        x += dropout(z, self.dropout, rng2, training)
        return x
