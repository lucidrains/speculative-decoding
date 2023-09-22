import math

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from beartype import beartype

from collections import namedtuple

from einops import rearrange

# constants

Cache = namedtuple('Cache', ['cached_kvs', 'embeds'])

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
    probs.scatter_(-1, ind, val)
    return probs

# rotary embeddings

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, offset = None):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq)
        t = rearrange(t, 'n -> 1 n')

        if exists(offset):
            t = t + offset[..., None]

        freqs = torch.einsum('b n , d -> b n d', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    seq_len = t.shape[-2]
    pos = rearrange(pos, 'b n d -> b 1 n d')
    pos = pos[..., -seq_len:, :]
    return t * pos.cos() + rotate_half(t) * pos.sin()

# different decoding strategies

@torch.no_grad()
def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = net(out, cache = cache, return_cache = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)

    return out[..., prompt_seq_len:]

# speculative decoding functions

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

@torch.no_grad()
def speculative_decoding(
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    prompt_seq_len, out, device = prompt.shape[-1], prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    assert prompt.shape[0] == 1, 'batched spec decoding not supported yet'

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    while out.shape[-1] < seq_len:

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        for _ in range(gamma):
            small_logits, small_cache = small_net(out, cache = small_cache, return_cache = True)
            small_logits = small_logits[:, -1]

            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample[..., None]), dim = -1)

            q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.stack(all_small_logits, dim = -2)

        # verify with larger network

        logits, cache = net(out, cache = cache, return_cache = True)

        logits = logits[..., -(gamma + 1):, :]
        logits = top_k(logits, thres = filter_thres)

        # prob and prob of small model (p(x) and q(x) in algorithm 1)

        prob = safe_div(logits, temperature).softmax(dim = -1)
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))
        n = accepted.item() # need to handle batched spec decoding

        total_accepted += n
        num_steps += 1

        if n < gamma:
            adjusted_prob = F.relu(prob[:, n] - small_prob[:, n])
            prob_next = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
            out = out[:, :-(gamma - n)]

        # adjust cache

        next_seq_len = out.shape[-1]
        cache = tuple(t[..., :next_seq_len, :] for t in cache)
        small_cache = tuple(t[..., :next_seq_len, :] for t in small_cache)

        # sample the additional token

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)

    return out[..., prompt_seq_len:], total_accepted / num_steps

@torch.no_grad()
def speculative_decoding_with_same_model(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    prompt_seq_len, out, device = prompt.shape[-1], prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    assert prompt.shape[0] == 1, 'batched spec decoding not supported yet'

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    while out.shape[-1] < seq_len:

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        for _ in range(gamma):
            small_logits, small_cache = net(out, cache = small_cache, return_cache = True, return_early_exit_only = True)
            small_logits = small_logits[:, -1]

            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample[..., None]), dim = -1)

            q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.stack(all_small_logits, dim = -2)

        # verify with larger network

        logits, cache = net(out, cache = cache, early_exit_cache = small_cache, return_cache = True, start_from_early_exit_hiddens = True)

        logits = logits[..., -(gamma + 1):, :]
        logits = top_k(logits, thres = filter_thres)

        # prob and prob of small model (p(x) and q(x) in algorithm 1)

        prob = safe_div(logits, temperature).softmax(dim = -1)
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))
        n = accepted.item() # need to handle batched spec decoding

        total_accepted += n
        num_steps += 1

        if n < gamma:
            adjusted_prob = F.relu(prob[:, n] - small_prob[:, n])
            prob_next = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
            out = out[:, :-(gamma - n)]

        # adjust cache

        next_seq_len = out.shape[-1]
        cache = tuple(t[..., :next_seq_len, :] for t in cache)
        small_cache = tuple(t[..., :next_seq_len, :] for t in small_cache)

        # sample the additional token

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)

    return out[..., prompt_seq_len:], total_accepted / num_steps

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
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cache = None,
        context_mask = None,
        rotary_emb = None
    ):
        h, device = self.heads, x.device

        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        cached_kv = torch.stack((k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

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
        ignore_index = -1,
        early_exit_layer = None,
        early_exit_extra_transformer_blocks = 0,
        detach_early_exit_hiddens = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.detach_early_exit_hiddens = detach_early_exit_hiddens
        self.early_exit_layer = early_exit_layer
        self.to_early_exit_logits = None
        self.early_exit_transformer_blocks = ModuleList([])

        if exists(early_exit_layer):
            for _ in range(early_exit_extra_transformer_blocks):
                self.early_exit_transformer_blocks.append(ModuleList([
                    CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

            self.to_early_exit_logits = nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        return_loss = False,
        return_cache = False,
        seq_start_pos_offset = None,
        cache = None,
        early_exit_cache = None,
        return_early_exit_only = False,
        start_from_early_exit_hiddens = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle seq start pos offset

        self_attn_kv_mask = None
        if exists(seq_start_pos_offset):
            batch, seq_len = x.shape[:2]
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long)
            self_attn_kv_mask = seq_range >= seq_start_pos_offset[..., None]

        # relative positional encoding

        rotary_emb = self.rotary_emb(x.shape[-2], offset = seq_start_pos_offset)

        # setup cache

        new_cached_kvs = []

        cache_kvs = cache_embeds = None

        if exists(cache):
            cache_kvs, cache_embeds = cache

        cache_kvs = default(cache_kvs, [])
        iter_cache_kvs = iter(cache_kvs)

        # handle if previous cached embedding layer from early exit layer passed in

        layers = self.layers

        if start_from_early_exit_hiddens:
            assert not return_early_exit_only and exists(early_exit_cache)
            early_exit_layer_index = self.early_exit_layer

            early_cache_kvs, cache_embeds = early_exit_cache

            cache_embeds_len = cache_embeds.shape[-2]

            assert cache_embeds_len <= x.shape[-2]

            early_exit_layers, layers = layers[:early_exit_layer_index], layers[early_exit_layer_index:]
            x = x[:, cache_embeds_len:]

            iter_early_cache_kvs = iter(early_cache_kvs)

            for ind, (attn, ff) in enumerate(early_exit_layers):
                residual = x
                attn_out, cached_kv = attn(x, context_mask = self_attn_kv_mask, rotary_emb = rotary_emb, cache = next(iter_early_cache_kvs, None))
                x = residual + attn_out

                new_cached_kvs.append(cached_kv)

                x = ff(x) + x

            x = torch.cat((cache_embeds, x), dim = -2)

        # if cache passed in, just use the last token

        if exists(cache) :
            num_tokens_keep = x.shape[-2] - cache_kvs.shape[-2]
            x = x[:, -num_tokens_keep:]

        early_exit_hiddens = None

        # main transformer body

        for ind, (attn, ff) in enumerate(layers):
            layer = ind + 1

            residual = x
            attn_out, cached_kv = attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
            x = residual + attn_out

            new_cached_kvs.append(cached_kv)

            x = ff(x) + x

            if layer == self.early_exit_layer:
                early_exit_hiddens = x

                if self.detach_early_exit_hiddens:
                    early_exit_hiddens = early_exit_hiddens.detach()

                for early_exit_attn, early_exit_ff in self.early_exit_transformer_blocks:
                    residual = x
                    attn_out, cached_kv = early_exit_attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
                    x = residual + attn_out

                    new_cached_kvs.append(cached_kv)

                    x = early_exit_ff(x) + x

                if return_early_exit_only:
                    break

        new_cached_kvs = torch.stack(new_cached_kvs)

        to_logits = self.to_logits if not return_early_exit_only else self.to_early_exit_logits

        logits = to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            if exists(cache_embeds):
                x = torch.cat((cache_embeds, x), dim = -2)

            return logits, Cache(new_cached_kvs, x)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        if not exists(self.to_early_exit_logits):
            return loss

        early_exit_logits = self.to_early_exit_logits(early_exit_hiddens)

        early_exit_loss = F.cross_entropy(
            rearrange(early_exit_logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, early_exit_loss
