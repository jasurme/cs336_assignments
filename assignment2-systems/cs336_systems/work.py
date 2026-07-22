import argparse
from collections.abc import Callable
import numpy as np
import timeit
parser = argparse.ArgumentParser()
from cs336_basics.model import *
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--context_length', default=512, type=int)
parser.add_argument('--d_model', type=int)
parser.add_argument('--d_ff', type=int)
parser.add_argument('--rope_theta', default=10000, type=int)
parser.add_argument('--num_layers', type=int)
parser.add_argument('--num_heads', type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--maxlrate', default=1e-3, type=int)
parser.add_argument('--minlrate', default=1e-4, type=int)
parser.add_argument('--warmupiters', default=200, type=int)
parser.add_argument('--coscycleiters', default=5000, type=int)
parser.add_argument('--device', default='mps')
parser.add_argument('--max_l2_norm', default=1, type=int)
# parser.add_argument('--optimizer', default='AdamW')

parser.add_argument('--num_warmups', type=int)
parser.add_argument('--num_trials', type=int)


args = parser.parse_args()



model = TransformerLM(args.vocab_size, args.context_length,args.num_layers,args.d_model, args.num_heads, args.d_ff,args.rope_theta ).to(args.device)


inputs = torch.randint(0, args.vocab_size, (args.batch_size,args.context_length)).to(args.device)
targets = torch.randint(0, args.vocab_size, (args.batch_size,args.context_length)).to(args.device)

adamw_optimizer = AdamW(model.parameters(), args.maxlrate)

for _ in range(args.num_warmups):
    model.forward(inputs)

torch.mps.synchronize()

times = []
for trial in range(args.num_trials):

    start = timeit.default_timer()
    logits = model.forward(inputs)
    logits = einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab")
    targets_new = einops.rearrange(targets, "batch seq -> (batch seq)")
    loss = cross_entropy(logits, targets_new)
    print(f'step: {trial} - loss: {loss.item()}')
    adamw_optimizer.zero_grad()
    loss.backward()


    torch.mps.synchronize()
    end = timeit.default_timer()



    times.append(end-start)


avg_time = sum(times)/len(times)

print('avg_time: ', avg_time)
