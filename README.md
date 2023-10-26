# Attention

*Python version:* 3.11.6

Run tests with `pytest`

Show logs with `pytest --log-level-cli=25` to show custom logs logged with `logging.log(25, "message")`
It is also set to `25` by default in `pyproject.toml`.

### TODO

- [x] Implement softmax
- [ ] Implement network
- [ ] Implement vanilla attention
- [ ] Implement multi headed attention
- [ ] Add tests to compare with pytorch implementations
- [ ] Windowed attention
- [ ] Windowed attention with recomputation
- [ ] Streaming LLM attention

#### Links

Stateful computation: https://jax.readthedocs.io/en/latest/jax-101/07-state.html

Efficient Streaming Attention: https://arxiv.org/abs/2309.17453

Softmax off by one: https://www.evanmiller.org/attention-is-off-by-one.html

ML Library in Jax: https://flax.readthedocs.io/en/latest/