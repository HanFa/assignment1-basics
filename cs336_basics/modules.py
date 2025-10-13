import torch
import torch.nn as nn
import numpy as np
from einops import einsum, reduce, rearrange
from typing import Iterable


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


class SiLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, torch.sigmoid(x), "..., ... -> ...")
        return out


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
        self.silu = SiLU()
        self.weight2 = _init_weights(d_model, self.d_ff)
        self.weight3 = _init_weights(self.d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.weight1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = self.silu(w1x)
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        seq_len = x.shape[-2]

        # slice from the r buffer
        if token_positions is not None:
            token_positions = token_positions.flatten()
        else:
            token_positions = torch.arange(seq_len)  # default to use arange if not specified

        rotary = self.r[token_positions, :, :]

        assert rotary.shape == (len(token_positions), self.d_k, self.d_k)
        assert len(token_positions) == seq_len

        rotated_x = einsum(rotary, x, "seq_len d_out d_in, ... seq_len d_in -> ... seq_len d_out")
        assert rotated_x.shape == x.shape

        return rotated_x


class Softmax(nn.Module):

    def __init__(self, dim: int):
        """
        Implement softmax layer
        """
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_x = torch.max(x, dim=self.dim, keepdim=True)  # prevent nan

        x = x - max_x.values
        ex = torch.exp(x)

        sum_ex = torch.sum(ex, dim=self.dim, keepdim=True)

        result = ex / sum_ex

        assert result.shape == x.shape

        return result


class ScaledDotProdAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        d_k, n_len, m_len = query.shape[-1], query.shape[-2], key.shape[-2]

        qk = einsum(query, key, "batch_size ... n_len d_k, batch_size ... m_len d_k -> batch_size ... m_len n_len")
        qk = rearrange(qk, "batch_size ... m_len n_len ->  batch_size ...  n_len m_len")

        if mask is None:
            mask = torch.tril(torch.full_like(qk, True, dtype=torch.bool))

        qk += torch.where(mask, 0.0, float('-inf'))
        assert qk.shape == mask.shape

        softmax = Softmax(dim=-1)

        s = softmax(qk / np.sqrt(d_k))

        attn = einsum(s, value, "batch_size ...  n_len m_len, batch_size ... m_len d_v -> batch_size ... n_len d_v")

        return attn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, theta: float = 0, max_seq_len: int = 0, apply_rope: bool = False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model

        self.apply_rope = apply_rope
        if apply_rope:
            self.theta = theta
            self.max_seq_len = max_seq_len
            self.rope = RotaryPositionalEmbedding(theta, self.d_k // num_heads, self.max_seq_len)

        def _proj(out_dim, in_dim, dtype=None, device=None):
            sigma = np.sqrt(2 / (in_dim + out_dim))
            w = nn.init.trunc_normal_(
                torch.empty(out_dim, in_dim, dtype=dtype, device=device),
                mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
            return nn.Parameter(w)

        self.q_proj = _proj(self.d_k, self.d_model, dtype=torch.float32)
        self.k_proj = _proj(self.d_k, self.d_model, dtype=torch.float32)
        self.v_proj = _proj(self.d_k, self.d_model, dtype=torch.float32)
        self.o_proj = _proj(self.d_model, self.d_k, dtype=torch.float32)

        self.attn_blocks = [ScaledDotProdAttention() for _ in range(num_heads)]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        wqx = einsum(self.q_proj, x, "d_k d_model, batch_size ... d_model -> batch_size ... d_k")
        wkx = einsum(self.k_proj, x, "d_k d_model, batch_size ... d_model -> batch_size ... d_k")
        wvx = einsum(self.v_proj, x, "d_k d_model, batch_size ... d_model -> batch_size ... d_k")

        attn_out = []

        d_k = self.d_k // self.num_heads

        for i in range(self.num_heads):
            q, k, v = wqx[..., i * d_k: (i + 1) * d_k], wkx[..., i * d_k: (i + 1) * d_k], wvx[...,
                                                                                          i * d_k: (i + 1) * d_k]

            if self.apply_rope:
                if token_positions is None:
                    token_positions = torch.arange(x.size(1), device=x.device)

                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)

            attn_out.append(
                self.attn_blocks[i](
                    query=q,
                    key=k,
                    value=v,
                    mask=None
                )
            )

        attn_out = torch.concat(attn_out, dim=-1)
        assert attn_out.shape[-1] == self.d_k

        mult_attn_out = einsum(self.o_proj, attn_out, "d_model d_k, batch_size ... d_k -> batch_size ... d_model")
        return mult_attn_out


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.block = nn.Sequential(
            RMSNorm(d_model),
            MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, apply_rope=True)
        )

        self.block2 = nn.Sequential(
            RMSNorm(d_model),
            SwiGLUFeedForward(d_model, d_ff)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x + self.block(x)
        out = z + self.block2(z)
        assert out.shape == x.shape
        return out


class TransformerLM(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, vocab_size: int,
                 num_layers: int):
        super().__init__()

        self.token_embedding = Embedding(vocab_size, d_model)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.output_embedding = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.output_embedding(x)  # returns un-normalized output logits
        return x


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)

    # Compute softmax probabilities
    max_logit = torch.max(logits, dim=1, keepdim=True)[0]
    logits_stable = logits - max_logit

    log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)

    # Take log and gather target probabilities
    log_probs = logits_stable - log_sum_exp
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Return negative mean log likelihood
    return -target_log_probs.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]

    norms: list[torch.Tensor] = []
    norms.extend([torch.linalg.vector_norm(grad) for grad in grads])
    grad_norm = torch.linalg.vector_norm(torch.stack(norms))
    if grad_norm > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad *= (max_l2_norm / (grad_norm + 1e-6))
