from base import Array, BaseModule
from transformer import Embedding
from attention import Linear, LayerNorm
from states import GPT2DenseState
from act import gelu_new, dropout

class GPT2(BaseModule):
    def __init__(self, vocab_size: int=50257, emb_size: int=768):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.context_len = 1024

        self.wte = Embedding(vocab_size, emb_size)
        self.wpe = Embedding(self.context_len, emb_size)

        
        self.lm_head = Linear(emb_size, vocab_size, bias=False)


# activation: gelu_new
# attention: GPT2Attention
class GPT2Block(BaseModule):
    def __init__(self, emb_size: int=768):
        super().__init__()

        self.emb_size = emb_size

        self.ln_1 = LayerNorm(emb_size)
        self.attn = GPT2Attention(emb_size)
        self.ln_2 = LayerNorm(emb_size)

        
class GPT2Attention(BaseModule):
    def __init__(self, emb_size: int=768):
        super().__init__()

        self.emb_size = emb_size


class GPT2Dense(BaseModule):
    def __init__(self, intermediate_size: int, emb_size: int=768):
        super().__init__()

        self.emb_size = emb_size # input features
        self.intermediate_size = intermediate_size

        self.c_fc = Linear(emb_size, self.intermediate_size)
        self.c_proj = Linear(self.intermediate_size, emb_size)

    def forward(
        self, 
        states: GPT2DenseState,
        x: Array["..., emb_size"],
        rng: Array,
        training: bool=False
    ) -> Array:

        # out.shape: (..., emb_size)
        x = self.c_fc(states.c_fc, x) 
        x = gelu_new(x)
        x = self.c_proj(states.c_proj, x)
        if training: x = dropout(x, 0.1, rng)
        out = x
        return out

