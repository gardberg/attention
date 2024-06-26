# Attention

Exploring attention and related concepts in Jax.


### Demonstrating quadratic time and memory scaling

<div style="display: flex; justify-content: space-between;">
    <img src="images/attention_time_scaling.png" width="400" height="250" />
    <img src="images/attention_memory_scaling.png" width="400" height="250" />
</div>

### Dense example

See `examples/example_dense.ipynb`.

<div style="text-align:center;">
    <img src="images/decision_boundary.gif" width="470" height="330" />
</div>

### Setup

*Python version:* 3.11.6

Run tests with `pytest`

Filter for a specific test to run with `pytest -k test_name`

Enable debug logs with `pytest --log-cli-level=DEBUG`

Run formatting with `black .`

### TODO

- [ ] Ensure split is used on all rng keys in functions

If the same key is split in two different places, does it result in the same subkeys?

- [x] Implement softmax
- [x] Implement network
- [x] Implement batch norm
- [x] Implement vanilla attention
- [x] Implement multi headed attention

- [x] Add tests to compare with pytorch implementations
- [x] Simplify logging
- [ ] Switch to __name__ logging
- [x] Switch to using single Array type for shape hinting

#### Transformer

- [x] Switch to using `poetry` for dependencies
- [x] Attention Masking
- [x] Layer Norm
- [x] Dropout
- [x] Positional Encoding
- [x] Tokenizer (tiktoken)
- [x] Encoder
- [x] Decoder
- [x] Transformer
- [x] Seq2Seq
- [ ] T5
  - [ ] Inference using random weights
  - [ ] Inference using t5-small weights

#### Improvements

- [x] RMSNorm
- [x] SwiGLU layer
- [x] GeGLU
- [x] Rotational Positional Encoding
- [x] K/V Cache
- [ ] Multi-Query Attention (https://arxiv.org/abs/1911.02150)
- [ ] Grouped Query Attention

- [ ] Vectorize stuff with vmap
- [ ] Windowed attention with recomputation
- [ ] Streaming LLM attention

#### Extra fun stuff!

- [x] Snake activation

#### Using Poetry

Make sure local venv is activated, which contains poetry:

`source .venv/Scripts/activate` - Windows

`poetry add <package>`

`poetry check` - check for errors in `pyproject.toml`

`poetry run pytest` - use local venv to run pytest

`poetry install` - install dependencies specified in `pyproject.toml`

Have added pytest config options to `pyproject.toml` so that they are set by default. E.g. `log_cli = true` 

### Links

Tree Map: https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html

Stateful computation: https://jax.readthedocs.io/en/latest/jax-101/07-state.html

Efficient Streaming Attention: https://arxiv.org/abs/2309.17453

Softmax off by one: https://www.evanmiller.org/attention-is-off-by-one.html

ML Library in Jax: https://flax.readthedocs.io/en/latest/

NN Example: https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

#### Transformers

Illustrated transformer: https://jalammar.github.io/illustrated-transformer/

Annotated transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html

Pytorch implementation https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L5101

Annotated pytorch implementation https://nn.labml.ai/transformers/mha.html
https://nn.labml.ai/transformers/models.html

Attention explained https://storrs.io/attention/

https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product

Self-attention Does Not Need O(n^2) Memory https://arxiv.org/abs/2112.05682

#### Modern Transformer

Reformer: The Efficient Transformer https://arxiv.org/abs/2001.04451

RoFormer: Enhanced Transformer with Rotary Position Embedding https://arxiv.org/abs/2104.09864

Fast Transformer Decoding: One Write-Head is All You Need https://arxiv.org/abs/1911.02150 (Multi-Query Attention)

Llama2: https://arxiv.org/abs/2307.09288

GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
