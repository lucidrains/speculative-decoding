import math

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from beartype import beartype

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention and feedforward

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        rotary_emb: RotaryEmbedding,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cache = None
    ):
        h, device = self.heads, x.device

        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        cached_kv = torch.stack((k, v))

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, cached_kv

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class Decoder(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ignore_index = -1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        rotary_emb = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.ignore_index = ignore_index

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        n, out = prompt.shape[-1], prompt.clone()

        sample_num_times = max(1, seq_len - prompt.shape[-1])

        cache = None

        for _ in range(sample_num_times):
            logits, cache = self.forward(out, cache = cache, return_cache = True)
            logits = logits[:, -1]

            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample[..., None]), dim = -1)

        return out[..., n:]

    def forward(
        self,
        x,
        return_loss = False,
        return_cache = False,
        cache = None
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # next cache

        new_cached_kvs = []

        # if cache passed in, just use the last token

        if exists(cache):
            assert not self.training
            num_tokens_keep = x.shape[-2] - cache.shape[-2]
            x = x[:, -num_tokens_keep:]

        cache = default(cache, [])
        iter_cache = iter(cache)

        for attn, ff in self.layers:
            residual = x
            attn_out, cached_kv = attn(x, cache = next(iter_cache, None))
            x = residual + attn_out

            new_cached_kvs.append(cached_kv)

            x = ff(x) + x

        new_cached_kvs = torch.stack(new_cached_kvs)

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, new_cached_kvs

        return F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )
