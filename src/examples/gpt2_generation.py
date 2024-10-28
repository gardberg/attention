"""
Example script of "streaming" generation from a GPT2 model.
"""
import sys
sys.path.append('../')

import argparse

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
    if prompt:
        input_ids = jnp.array(tokenizer(prompt)["input_ids"])
    else:
        input_ids = None
    rng = jax.random.PRNGKey(0)
    
    current_text = prompt
    # Clear the previous empty line and start on the current line
    if prompt != "":
        print("\033[A\033[K" + current_text, end="", flush=True)
    else:
        print()
    
    for next_token_id in model.generate_tokens(pretrained_state, rng, input_ids, max_new_tokens=100):
        next_text = tokenizer.decode(next_token_id)
        next_text = next_text.replace('\n', ' ')
        current_text += next_text
        # Clear the line with \033[K before printing the new text
        print("\033[K" + current_text, end="\r", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prompt", action="store_true", help="Generate without prompting for input")
    args = parser.parse_args()

    if args.no_prompt:
        generate("")
        print()
    else:
        while True:
            print("\nEnter prompt (or Ctrl+C to exit):\n\n", end="", flush=True)
            prompt = sys.stdin.readline().strip()
            if prompt:
                generate(prompt)
                print()  # Add a newline after generation is complete
