import math

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

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

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    seq_len = t.shape[-2]
    pos = pos[-seq_len:, :]
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

        self.norm = nn.RMSNorm(dim)

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
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        cached_kv = torch.stack((k, v), dim = 1)

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
        nn.RMSNorm(dim),
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
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        start_tokens = None,
        return_loss = False,
        return_cache = False,
        seq_start_pos = None,
        cache = None
    ):
        has_start_tokens = exists(start_tokens)

        start_token_len = 0
        if exists(start_tokens):
            if start_tokens.ndim == 2:
                start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

            start_token_len = start_tokens.shape[-2]

        if return_loss:
            x, labels = x[:, start_token_len:-1], x[:, 1:]

        x = self.token_emb(x)

        if exists(start_tokens):
            x = torch.cat((start_tokens, x), dim = 1)

        # handle seq start pos offset

        self_attn_kv_mask = None
        if exists(seq_start_pos):
            batch, seq_len = x.shape[:2]
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long)
            self_attn_kv_mask = seq_range >= seq_start_pos[..., None]

        # relative positional encoding

        rotary_emb = self.rotary_emb(x.shape[-2])

        # setup cache

        new_cached_kvs = []

        cache_kvs = cache_embeds = None

        if exists(cache):
            cache_kvs, cache_embeds = cache

        if exists(cache_kvs):
            iter_cache_kvs = iter(cache_kvs.unbind(dim = 1))
        else:
            iter_cache_kvs = iter([])

        # if cache passed in, just use the last token

        if exists(cache):
            num_tokens_keep = x.shape[-2] - cache_kvs.shape[-2]
            x = x[:, -num_tokens_keep:]

        # main transformer body

        for ind, (attn, ff) in enumerate(self.layers):
            layer = ind + 1

            residual = x
            attn_out, cached_kv = attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
            x = residual + attn_out

            new_cached_kvs.append(cached_kv)

        new_cached_kvs = torch.stack(new_cached_kvs, dim = 1)

        logits = self.to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, Cache(new_cached_kvs, x)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, Cache(new_cached_kvs, x)

class ModelWithProphetWrapper(Module):
    def __init__(
        self,
        model: Decoder,
        prophet: Decoder,
        prophet_train_length = 8,  # should be greater than spec decoding gamma, as main model cache embedding is one step behind
        detach_model_embed_for_prophet = False,
        num_leading_start_tokens = 1
    ):
        super().__init__()
        self.model = model
        self.prophet = prophet

        model_prophet_same_dim = model.dim == prophet.dim
        self.to_prophet_start_token = nn.Identity() if model_prophet_same_dim else nn.Linear(model.dim, prophet.dim, bias = False)

        assert num_leading_start_tokens >= 1
        self.num_leading_start_tokens = num_leading_start_tokens

        self.prophet_train_length = prophet_train_length
        self.detach_model_embed_for_prophet = detach_model_embed_for_prophet

    def forward(self, x):
        num_start_tokens = self.num_leading_start_tokens
        batch, seq_len, device = *x.shape, x.device
        prophet_seq_len = self.prophet_train_length
        assert seq_len >= prophet_seq_len

        total_loss = 0.

        main_loss, (cached_kvs, embeds) = self.model(x, return_loss = True)

        total_loss = total_loss + main_loss

        if self.detach_model_embed_for_prophet:
            embeds = embeds.detach()

        prophet_start_tokens = self.to_prophet_start_token(embeds)

        batch_arange = torch.arange(batch, device = device, dtype = torch.long)
        prophet_seq_arange = torch.arange(prophet_seq_len, device = device, dtype = torch.long)

        num_seq_train_prophet = seq_len - prophet_seq_len - (num_start_tokens - 1)

        offsets = torch.arange(num_seq_train_prophet, device = device, dtype = torch.long)

        prophet_input = x[
            batch_arange[:, None, None],
            offsets[..., None] + prophet_seq_arange
        ]

        prophet_input = rearrange(prophet_input, '... n -> (...) n')

        start_tokens_arange = torch.arange(num_start_tokens, device = device, dtype = torch.long)

        prophet_start_tokens = prophet_start_tokens[
            batch_arange[:, None, None],
            offsets[..., None] + start_tokens_arange
        ]

        prophet_start_tokens = rearrange(prophet_start_tokens[:, :num_seq_train_prophet], 'b n l d -> (b n) l d')

        prophet_loss, _ = self.prophet(prophet_input, start_tokens = prophet_start_tokens, return_loss = True)

        total_loss = total_loss + prophet_loss

        return total_loss, (main_loss, prophet_loss)

# speculative decoding functions

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

@torch.no_grad()
def speculative_decoding_with_prophet_model(
    net: ModelWithProphetWrapper,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    # extract model, prophet, and model to prophet (if their model dimensions differ)

    model = net.model
    to_prophet_start_token = net.to_prophet_start_token
    prophet = net.prophet
    num_start_tokens = net.num_leading_start_tokens

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device

    if (seq_len - prompt_seq_len) <= 0:
        return prompt, None

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

    # sample the first token from the main model

    for _ in range(max(1, num_start_tokens - prompt_seq_len)):
        logits, cache = model(out, cache = cache, return_cache = True)
        logits = logits[:, -1:]
        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)
        out = torch.cat((out, sample), dim = -1)
        seq_lens += 1

    # now we have the first cached embedding to use as the prophet network start token for the speculative sampling

    _, embeds = cache
    next_prophet_start_tokens = to_prophet_start_token(embeds[:, -num_start_tokens:])

    while (seq_lens < seq_len).any():

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        small_cache = None
        num_tokens = 2  # the main model embeddings is 1 step behind the main sequence

        for _ in range(gamma):
            small_logits, small_cache = prophet(
                out[..., -num_tokens:],
                start_tokens = next_prophet_start_tokens,
                cache = small_cache,
                return_cache = True
            )

            small_logits = small_logits[:, -1:]

            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample), dim = -1)

            seq_lens += 1
            num_tokens += 1

            q_sampled_out.append(rearrange(sample, '... -> ... 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.cat(all_small_logits, dim = -2)

        # verify with larger network

        logits, cache = model(
            out,
            cache = cache,
            return_cache = True,
            seq_start_pos = out.shape[-1] - seq_lens
        )

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

        total_accepted += accepted.float().mean()
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, 'b -> b 1')
        accepted.clamp_(max = gamma - 1)

        adjusted_prob = F.relu(prob[batch_range, accepted] - small_prob[batch_range, accepted])
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
        adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

        prob_next = torch.where(
            rearrange(has_rejected, '... -> ... 1'),
            adjusted_prob,
            prob_next
        )

        # do a bunch of slicing and align everything to the right, including kv caches

        max_num_rejected = num_rejected.amax()

        curr_len = out.shape[-1]
        seq_lens -= num_rejected
        max_seq_len = seq_lens.amax()

        seq_arange = torch.arange(max_seq_len, device = device, dtype = torch.long) + (curr_len - max_seq_len)

        seq_offset_indices = seq_arange - num_rejected[..., None]

        cached_kv, embed = cache

        if batch > 1:
            out = out[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b ... n d -> b n ... d')
            cached_kv = cached_kv[batch_range, seq_offset_indices]
            cached_kv = rearrange(cached_kv, 'b n ... d -> b ... n d')
        else:
            out = out[..., :max_seq_len]
            cached_kv = cached_kv[..., :max_seq_len, :]

        cache = (cached_kv, None)

        # sample the additional token, one of the tricks in the paper to better bound the worst case

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)
        seq_lens += 1

        next_prophet_start_tokens = to_prophet_start_token(embeds[:, -num_start_tokens:])

    # now left align

    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value = pad_id)

    seq_len_range = torch.arange(seq_len, device = device, dtype = torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps
