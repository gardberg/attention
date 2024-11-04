from IPython.display import HTML, display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import jax.numpy as jnp

from base import Array


def visualize_token_probabilities(text: str, probabilities: Array, tokenizer):
    """
    Visualize token-level probabilities using color coding

    Args:
        text: Text to visualize
        probabilities: Array of probabilities for each token in tokenized text
        tokenizer: Tokenizer to tokenize the text
    """

    # Tokenize input text
    tokens = tokenizer.encode(text)

    # Verify tokens and probabilities match
    assert len(tokens) == len(
        probabilities
    ), f"Number of tokens ({len(tokens)}) does not match number of probabilities ({len(probabilities)})"

    # Calculate per-token perplexity
    perplexities = jnp.array([2 ** (-jnp.log2(p)) for p in probabilities])

    # Normalize for coloring (using log scale for better visualization)
    log_perplexities = jnp.log2(perplexities)
    min_perp, max_perp = jnp.min(log_perplexities), jnp.max(log_perplexities)
    normalized = (log_perplexities - min_perp) / (max_perp - min_perp)

    # Create HTML with color gradient (red = high perplexity, blue = low perplexity)
    html_parts = []

    # Get the original text for each token
    token_texts = []
    current_pos = 0
    decoded_text = text

    for token in tokens:
        # Decode the token to get its text representation
        token_text = tokenizer.decode([token])
        # Find where this token appears in the original text
        token_pos = decoded_text.find(token_text, current_pos)
        if token_pos != -1:
            token_texts.append(decoded_text[token_pos : token_pos + len(token_text)])
            current_pos = token_pos + len(token_text)

    for token_text, norm_value in zip(token_texts, normalized):
        # Create color (blue to red gradient)
        color = mcolors.to_hex(plt.cm.RdYlBu_r(norm_value))
        html_parts.append(f'<span style="color: {color}">{token_text}</span>')

    html = "".join(html_parts)
    display(HTML(html))
