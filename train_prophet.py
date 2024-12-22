import gzip
import random
import tqdm
import numpy as np
import time
from functools import wraps, partial

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.cuda import synchronize, Event
from torch.utils.data import DataLoader, Dataset

timer = partial(Event, enable_timing = True)

from speculative_decoding.speculative_decoding_with_prophet import (
    Decoder,
    ModelWithProphetWrapper,
    base_decoding,
    speculative_decoding_with_prophet_model
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
PRIME_LENGTH = 128
GENERATE_EVERY = 100
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5
TRAIN_PROPHET = True

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def benchmark(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms
    return inner

# instantiate transformer

device = torch.device(DEVICE_STR)

model = Decoder(
    num_tokens = 256,
    dim = 512,
    depth = 10
)

prophet = Decoder(
    num_tokens = 256,
    dim = 512,
    depth = 2
)

model_and_prophet = ModelWithProphetWrapper(
    model,
    prophet,
    prophet_train_length = GAMMA + 2,
    num_leading_start_tokens = 2,
    detach_model_embed_for_prophet = False   # train end to end, shouldn't hurt (although benefits is dubious) given ProphetNet paper - of course, trying to get to the bottom of the benefits in spec decoding setting here
).to(device)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))

# optimizer

params = model_and_prophet.parameters() if TRAIN_PROPHET else model.parameters()

optim = Adam(params, lr = LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model_and_prophet.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        total_loss, (loss, prophet_loss) = model_and_prophet(data)

        (total_loss / GRAD_ACCUM_EVERY).backward()

    print(f"loss: {loss.item():.3f}\tprophet loss: {prophet_loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model_and_prophet.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % GENERATE_EVERY == 0:
        model_and_prophet.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        prompt = inp[None, ...]

        sampled, base_decode_elapsed = benchmark(base_decoding)(model, prompt, GENERATE_LENGTH)

        (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding_with_prophet_model)(model_and_prophet, prompt, GENERATE_LENGTH, GAMMA)

        base_decode_output = decode_tokens(sampled[0])
        spec_decode_output = decode_tokens(spec_decode_sampled[0])

        print("\nbase decoding:\n\n", base_decode_output, "\n")
        print("\nspec decoding:\n\n", spec_decode_output, "\n")

        print(f'base decoding in: {base_decode_elapsed:.3f}ms\n')
        print(f'spec decoding in: {spec_decode_elapsed:.3f}ms\n')
        print(f'average num accepted: {num_accepted:.1f} / {GAMMA}\n')
