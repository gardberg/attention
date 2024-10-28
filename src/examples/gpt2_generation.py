"""
Example script of "streaming" generation from a GPT2 model.
"""
import sys
sys.path.append('../')

import jax
import jax.numpy as jnp
from models.gpt2 import GPT2
from states import to_jax_state

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Copy pretrained weights
torch_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

pretrained_state = to_jax_state(torch_model)

model = GPT2()

def generate(prompt: str):
    input_ids = jnp.array(tokenizer(prompt)["input_ids"])
    rng = jax.random.PRNGKey(0)
    
    current_text = prompt
    # Move up one line with \033[A, then use \r to move to start of line
    # Only do this once at the start
    print("\033[A\r" + current_text, end="", flush=True)
    
    for next_token_id in model.generate_tokens(pretrained_state, input_ids, rng, max_new_tokens=100):
        current_text += tokenizer.decode(next_token_id)
        # Just use \r to return to start of the same line
        print("\r" + current_text, end="", flush=True)


if __name__ == "__main__":
    while True:
        print("\nEnter prompt (or Ctrl+C to exit):\n\n", end="", flush=True)
        prompt = sys.stdin.readline().strip()
        if prompt:
            generate(prompt)
            print()  # Add a newline after generation is complete
