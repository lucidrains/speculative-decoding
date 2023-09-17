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
    probs.scatter_(-1, ind, val)
    return probs

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
        cache = cache[..., :next_seq_len, :]
        small_cache = small_cache[..., :next_seq_len, :]

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
        weight_tie_layers = False,
        ignore_index = -1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        rotary_emb = RotaryEmbedding(dim = dim_head)

        attn = None
        ff = None

        for _ in range(depth):

            if not weight_tie_layers or not (exists(attn) and exists(ff)):
                attn = CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb)
                ff = FeedForward(dim = dim, mult = ff_mult)

            self.layers.append(ModuleList([attn, ff]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.ignore_index = ignore_index

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
