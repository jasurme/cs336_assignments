


import torch
import itertools
import einops
import math
d_model = [16, 32, 64, 128]
seq_len = [256, 1024, 4096, 8192, 16384]
device = 'cuda'
batchsize = 8
import gc
import timeit
from collections import defaultdict

# Q = torch.randn((batchsize,seql,dm),device=device, requires_grad=True )
# K = torch.randn((batchsize,seql,dm),device=device, requires_grad=True)
# V = torch.randn((batchsize,seql,dm),device=device, requires_grad=True)
def runattention(batchsize, seql, dm, Q, K, V, mask):



    scores = einops.einsum(Q, K, "... tokens_q d_q, ... tokens_k d_q -> ... tokens_q tokens_k")
    scores = scores.masked_fill(~mask,float("-inf"))
    after_softmax = torch.softmax(scores/math.sqrt(dm), dim=-1)
    return einops.einsum(after_softmax, V, "... tokens_q tokens_k, ... tokens_k d_v -> ... tokens_q d_v")

times = defaultdict(list)

for dm, seql in itertools.product(d_model, seq_len):
    Q = torch.randn((batchsize,seql,dm),device=device, requires_grad=True)
    K = torch.randn((batchsize,seql,dm),device=device, requires_grad=True)
    V = torch.randn((batchsize,seql,dm),device=device, requires_grad=True)
    mask = torch.tril(torch.ones(seql, seql, device=device)) ==1
    try:
    #warmup
        for i in range(5):
            out = runattention(batchsize, seql, dm, Q, K, V, mask)
            torch.cuda.synchronize()
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            Q.grad = None
            K.grad = None
            V.grad = None

        for i in range(100):
            start = timeit.default_timer()
            out = runattention(batchsize, seql, dm, Q, K, V, mask)
            torch.cuda.synchronize()

            end = timeit.default_timer()
            avgtime = end-start
            times[f'dm{dm}seql{seql}'].append(avgtime)
            print(f'forward avgtime for it {i}: dm{dm}seql{seql}: {avgtime}')
            memoryallocated = torch.cuda.memory_allocated()
            times[f'memoryallocateddm{dm}seql{seql}'].append(memoryallocated)
            print(f'memoryallocated before backward {i}:dm{dm}seql{seql}: {memoryallocated}')
            loss = out.sum()
            start_b= timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end_b= timeit.default_timer()
            avgtime_b = end_b-start_b
            times[f'backwarddm{dm}seql{seql}'].append(avgtime_b)
            print(f'backward avgtime for it {i}:dm{dm}seql{seql} : {avgtime_b}')
            Q.grad = None
            K.grad = None
            V.grad = None

        del Q, K, V, out, loss, mask
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Error for dm{dm}seql{seql}: {e}')
        Q.grad = None
        K.grad = None
        V.grad = None
        del Q, K, V, mask
        gc.collect()
        torch.cuda.empty_cache()
        continue

# multihead = einops.rearrange(heads, "... heads seq d_v -> ... seq (heads d_v)")





import statistics

for dm, seql in itertools.product(d_model, seq_len):
    try:
        mean = statistics.fmean(times[f"memoryallocateddm{dm}seql{seql}"])
        times[f"memoryallocateddm{dm}seql{seql}"] = mean
        print(f'backward mean time for dm{dm}seql{seql}: {mean}')
    except:
        print(f'skipping dm{dm}seql{seql}')


import csv


with open("times2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)


    writer.writerow(["Configurations", "(mean) value"])

   
    for key, value in times.items():
        writer.writerow([key, value])

print("CSV file created successfully!")
