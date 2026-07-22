
d_model = 1024
d_ff = 4096
num_layers = 24
num_heads = 1
use_mixed_precision = True
dtype_mixed = torch.bfloat16
# small: 768 3072 12 12
# medium: 1024 4096 24 16
# large: 1280 5120 36 20

vocab_size = 10000
context_length = 512
rope_theta = 10000
batch_size = 8                  # 4
maxlrate = 1e-3
minlrate = 1e-4
warmupiters = 200
coscycleiters = 5000

device = "cuda" if torch.cuda.is_available() else "cpu"
max_l2_norm = 1

num_warmups = 3
num_trials = 10


import numpy as np
import timeit
import torch
import einops
from contextlib import nullcontext
torch.cuda.memory._record_memory_history(max_entries=1000000)
def get_ctx():
        return torch.autocast(device_type=device,dtype=dtype_mixed) if use_mixed_precision else nullcontext()
model = TransformerLM(
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    d_ff,
    rope_theta
).to(device)



inputs = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
targets = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

adamw_optimizer = AdamW(model.parameters(), maxlrate)


# for _ in range(num_warmups):
#     with torch.no_grad():
#         with get_ctx():
#             model.forward(inputs)

if device == "cuda":
    torch.cuda.synchronize()
elif device == "mps":
    torch.mps.synchronize()


times = []

for trial in range(num_trials):

    # start = timeit.default_timer()
    # with torch.no_grad():
    with get_ctx():
        adamw_optimizer.zero_grad(set_to_none=True)

        logits = model.forward(inputs)
        # del logits

        logits = einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab")
        targets_new = einops.rearrange(targets, "batch seq -> (batch seq)")

        loss = cross_entropy(logits, targets_new)



    # start = timeit.default_timer()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
    # start = timeit.default_timer()
    adamw_optimizer.step()



    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # end = timeit.default_timer()

    # times.append(end - start)

torch.cuda.memory._dump_snapshot("fullcontext512.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
# avg_time = sum(times) / len(times)
# print("avg_time:", avg_time)

# print(times)




    