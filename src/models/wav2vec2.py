from base import BaseModule, Array
from conv import Conv1d
from states import (
    ConvLayerBlockState,
    FeatureExtractorState,
    FeatureProjectionState,
    ConvPosEmbeddingState,
    ConvPosTransformerState,
    Conv1dState,
)
from act import gelu, dropout
from attention import LayerNorm, Linear
from transformer import EncoderLayer
from log_utils import logger

from typing import Optional, Tuple, List
import jax.numpy as jnp
import jax



class Wav2Vec2(BaseModule):
    def __init__(self):
        super().__init__()

        self.sample_rate = 16000

    def forward(self):
        pass

        # Raw waveform ->

        # CNN Feature Extractor

        # Transformer Encoder

        # Quantization (?)

        # -> Embeddings

        # Classification Head

        # -> Logits


class ConvPosEmbedding(BaseModule):

    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        # no weight normalization during inference
        self.conv = Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        # used to remove the last feature if kernel_size is even
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(
        self,
        state: ConvPosEmbeddingState,
        x: Array["batch_size, n_frames, n_features"],
        normalize: bool = False,
    ) -> Array["batch_size, n_frames, n_features"]:
        x = x.transpose(0, 2, 1)

        # assume we are using pre-normalized weights
        if normalize: state = self.normalize_weights(state)
        x = self.conv(state.conv, x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = gelu(x)
        x = x.transpose(0, 2, 1)
        return x

    def normalize_weights(self, state: ConvPosEmbeddingState) -> ConvPosEmbeddingState:
        weights = state.conv.weight
        g = state.g
        norm = jnp.linalg.norm(weights, ord=2, axis=-1, keepdims=True)

        weights_normed = weights / norm * g
        return ConvPosEmbeddingState(
            conv=Conv1dState(weight=weights_normed, bias=state.conv.bias), g=g
        )


class ConvLayerBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[BaseModule] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.layer_norm = layer_norm

    def forward(
        self,
        state: ConvLayerBlockState,
        x: Array["batch_size, in_channels, in_length"],  # in_length: in_frame
        length: Optional[Array["batch_size,"]] = None,  # length of each sequence
    ) -> Tuple[
        Array["batch_size, out_channels, out_length"], Optional[Array["batch_size,"]]
    ]:

        x = self.conv(state.conv, x)
        if self.layer_norm is not None:
            x = self.layer_norm(state.layer_norm, x)
        x = gelu(x)

        # calculate resulting length
        if length is not None:
            length = (length - self.kernel_size) // self.stride + 1
            length = jnp.maximum(
                length, jnp.zeros_like(length)
            )  # make sure we dont get negative lengths

        return x, length


class FeatureExtractor(BaseModule):
    def __init__(self, conv_layers: List[ConvLayerBlock]):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(
        self,
        state: FeatureExtractorState,
        x: Array["batch_size, time"],
        length: Optional[Array["batch_size,"]] = None,
    ) -> Tuple[
        Array["batch_size, n_frames, n_features"], Optional[Array["batch_size,"]]
    ]:

        # (batch_size, in_channels=1, time)
        x = x.reshape(x.shape[0], 1, -1)

        for i, layer in enumerate(self.conv_layers):
            x, length = layer(state.conv_layers[i], x, length)

        x = x.transpose(0, 2, 1)

        return x, length


class Encoder(BaseModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.feature_projection = FeatureProjection(in_features, out_features)
        self.transformer = ConvPosTransformer()


class FeatureProjection(BaseModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.layer_norm = LayerNorm(in_features)
        self.projection = Linear(in_features, out_features, bias=True)

    def forward(
        self,
        state: FeatureProjectionState,
        x: Array["batch_size, n_frames, in_features"],
        rng: Array,
        training: bool = False
    ) -> Array["batch_size, n_frames, out_features"]:
        x = self.layer_norm(state.layer_norm, x)
        x = self.projection(state.projection, x)
        x = dropout(x, 0.1, rng, training)
        return x


# layer norm first
class ConvPosTransformer(BaseModule):
    def __init__(self, n_layers: int, emb_size: int = 768):
        super().__init__()

        self.n_layers = n_layers
        self.pos_conv_embed = ConvPosEmbedding(
            embed_dim=emb_size, kernel_size=128, groups=16
        )
        self.layer_norm = LayerNorm(emb_size)
        self.layers = [
            EncoderLayer(
                emb_size=emb_size,
                d_ff=emb_size * 4,
                n_heads=12,
                dropout=0.1,
                layer_norm_first=False,
                ff_activation=gelu,
                attn_out_bias=True,
                attn_qk_bias=True,
            )
            for _ in range(n_layers)
        ]

    def forward(
        self,
        state: ConvPosTransformerState,
        x: Array["batch_size, seq_len, emb_size"],
        rng: Array,
        training: bool = False,
    ) -> Array["batch_size, seq_len, emb_size"]:
        rngs = jax.random.split(rng, self.n_layers + 1)

        # preprocess
        x += self.pos_conv_embed(state.pos_conv_embed, x)
        x = self.layer_norm(state.layer_norm, x)
        x = dropout(x, 0.1, rngs[0], training)
        # transpose to encoder layer input shape (seq_len, batch_size, emb_size)
        x = x.transpose(1, 0, 2)
        for i, layer in enumerate(self.layers):
            x = layer(state.layers[i], x, rngs[i + 1])
        x = x.transpose(1, 0, 2)  # (batch_size, seq_len, emb_size)
        return x

    def get_intermediate_outputs(
        self,
        state: ConvPosTransformerState,
        x: Array["batch_size, seq_len, emb_size"],
        rng: Array,
        num_layers: int = None,
        training: bool = False,
    ) -> List[Array]:
        rngs = jax.random.split(rng, self.n_layers + 1)
        outputs = []
        x += self.pos_conv_embed(state.pos_conv_embed, x)

        x = self.layer_norm(state.layer_norm, x)

        x = dropout(x, 0.1, rngs[0], training)
        x = x.transpose(1, 0, 2)
        for i, layer in enumerate(self.layers):
            logger.debug(f"Layer {i}")
            x = layer(state.layers[i], x, rngs[i + 1])
            outputs.append(x)
            if num_layers is not None and len(outputs) >= num_layers:
                break
        return outputs
