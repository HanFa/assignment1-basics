import torch
import torch.nn as nn
import numpy as np
from einops import einsum


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        sigma = np.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(nn.init.trunc_normal_(
            tensor=torch.empty(out_features, in_features, dtype=dtype),
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")

        assert y.shape == (*x.shape[:-1], self.out_features)
        return y


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """

        Args:
            num_embeddings: size of the vocab
            embedding_dim: dimension of the embedding modules, aka d_model
            device: device to store parameters
            dtype: data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        sigma = np.sqrt(2 / (num_embeddings + embedding_dim))
        self.indexing = nn.Parameter(nn.init.trunc_normal_(
            tensor=torch.empty(num_embeddings, embedding_dim, dtype=dtype),
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )).to(device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        y = self.indexing[token_ids]

        assert y.shape == (*token_ids.shape, self.embedding_dim)
        return y


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """

        Args:
            d_model: hidden dimension of the model
            eps: epsilon value for numerical stability
            device: device to store parameters
            dtype: data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(
            torch.ones(d_model)
        ).to(device).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        sq_sum_x = einsum(x, x, "... d_model, ... d_model -> ...")
        rms_x = torch.sqrt(sq_sum_x / self.d_model + self.eps)

        assert rms_x.shape == x.shape[:-1]

        x_over_rms = einsum(x, 1 / rms_x, "... d_model, ... -> ... d_model")
        assert x_over_rms.shape == x.shape

        rms_norm = einsum(x_over_rms, self.weights, "... d_model, d_model -> ... d_model")
        assert rms_norm.shape == x.shape

        return rms_norm.to(in_dtype)


class SwiGLUFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        def _init_weights(out_dim, in_dim):
            sigma = np.sqrt(2 / (in_dim + out_dim))
            return nn.Parameter(nn.init.trunc_normal_(
                tensor=torch.empty(out_dim, in_dim, dtype=dtype),
                mean=0,
                std=sigma,
                a=-3 * sigma,
                b=3 * sigma,
            ))

        self.d_ff = d_ff
        self.weight1 = _init_weights(self.d_ff, d_model)
        self.weight2 = _init_weights(d_model, self.d_ff)
        self.weight3 = _init_weights(self.d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.weight1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = einsum(w1x, torch.sigmoid(w1x), "..., ... -> ...")
        w3x = einsum(self.weight3, x, "d_ff d_model, ... d_model -> ... d_ff")
        dot_product = einsum(silu, w3x, "..., ... -> ...")
        swiglu = einsum(self.weight2, dot_product, "d_model d_ff, ... d_ff -> ... d_model")

        return swiglu


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """

        Args:
            theta: the theta value for RoPE
            d_k: dimension of query and key vectors
            max_seq_len: max sequence length that will be inputted
            device: device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = d_k

        r = torch.zeros((max_seq_len, d_k, d_k))

        for index in range(max_seq_len):
            for k in range(d_k // 2):
                angle = float(index) / theta ** (2 * k / d_k)
                r[index, 2 * k: 2 * k + 2, 2 * k: 2 * k + 2] = torch.Tensor([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])

        assert r.shape == (max_seq_len, d_k, d_k)

        self.register_buffer('r', r, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        # slice from the r buffer
        rotary = self.r[token_positions, :, :]

        assert rotary.shape == (len(token_positions), self.d_k, self.d_k)

        rotated_x = einsum(rotary, x, "seq_len d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        assert rotated_x.shape == x.shape

        return rotated_x
