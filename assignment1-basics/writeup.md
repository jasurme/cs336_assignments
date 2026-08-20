## Problem (unicode1): Understanding Unicode

a) What Unicode character does chr(0) return?

- '\x00', null character

b) How does this character’s string representation (__repr__()) differ from its printed
representation?

- it prints nothing due to being non-printable control character while string repr prints "'\\\x00'" as memory

c) What happens when this character occurs in text?

- when terminal/interpreter faces this character, since it is non-printable control character that isnt served to be printed, but its purpose is to serve other purposes and so interpreter identifies it and skips this control character

## Problem (unicode2): Unicode Encodings (3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

- it is more efficient since utf-8 is dynamic. for example, since utf-32 gives 4 byte memory for any character, standard ascii 'a' gets 4 byte memory, and 3 bytes will be wasted as `0x00` (zero padding)

b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string
into a Unicode string. Why is this function incorrect? Provide an example of an input byte
string that yields incorrect results(Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong
produces incorrect output, with a one-sentence explanation of why the function is incorrect.)

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

- incorrect example is '간'. problem is it is decoding byte by byte. to decode characters that are longer than one byte(e.g. 2-4 bytes), partial decoding doesnt make sure and gives error

c) Give a two-byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation

- 'Д'. it is encoded as b'\xd0\x94' in utf-8 thats why when decoded with example function byte by byte, it is incorrect

## Problem (train_bpe): BPE Tokenizer Training (15 points)

- Training time: 221.61 seconds
  Peak memory (tracemalloc, main process): 139.4 MB
  Peak RSS (OS, whole process): 208.1 MB

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

a, b) compression ratio for tinystories sample: 4.039039039039039
throughput: 745,647 bytes/sec
pile dataset: 330.0 hours  (13.75 days)

- macbook m1 air was used
  - compression ratio for openweb sample: **3.1724415793714744 (with tinystores tokenizer because on M1 air, it couldn't do it)**

## Problem (transformer_accounting): Transformer LM resource accounting (5 points)

- number of trainable parameters: **1,640,452,800 = 1.64B**

vocab_size  ***d_m + num_layers*  * (2*d_m + 4 * d_m *d_m + 3 * d_ff * d_m) + d_m

- using single-precision floating point, 1,640,452,800 * 4 = **6,561,811,200**
- FLOPs:

```
rmsnorm = 4 * d_m * L
QKV = 3 * (2*L*d_m*d_m)
scores = 2*L*L*d_m
attention = 2*L*L*d_m
O_out = 2*L*d_m*d_m
FFN = 3 * (2*d_ff*d_m*L)
one_block = QKV + scores + attention + O_out + FFN
blocks = num_layers * one_block
lm_head = 2 * vocab_size * L* d_m
```

- total forward FLOPs - **approx. 3.52T flops**
- which parts of the model require the most FLOPs:  if we consider individually, LM_head linear also requires a lot of compute, but if num_layers are big, compute of total blocks overshadow it. inside transformer block. FFN also eats much compute

| Component                 |   GPT-2 Small |    GPT-2 Medium |     GPT-2 Large |              XL |
| ------------------------- | ------------: | --------------: | --------------: | --------------: |
| RMSNorm (non-trainable)   |          3.1M |            4.2M |            5.2M |            6.6M |
| QKV (d²)                 |          3.6B |            6.4B |           10.1B |           15.7B |
| Scores (d)                |          1.6B |            2.1B |            2.7B |            3.4B |
| Attention (d)             |          1.6B |            2.1B |            2.7B |            3.4B |
| Output Projection (d²)   |          1.2B |            2.1B |            3.6B |            5.2B |
| FFN (dₘ, d_ff)           |           20B |             27B |             33B |           42.2B |
| **One Block Total** | **28B** | **39.7B** | **52.1B** | **69.9B** |
| LM Head (d, V)            |           79B |            105B |            131B |            164B |

vocabsize=50257, contextlength = 1024. d_ff = 4288

- when d_model model size increases, QKV increases quadratically because it is 6 * contextlength * d^2. Output Projection too. others largely increase linearly. but because other hyperparaemters like vocabsize is large,, for example, FFN becomes larger
- XL .context length to 16,384. How does the total FLOPs for one forward pass change. 16x. ? it increases 16x2.46=39.3x. scores and attention increases double, quadratically because they have L^2

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)

- as i increase learning rate 1 -> 1e1, 1e2, loss decreased faster, so decay faster. but when i set to 1000, it went up, to infinity

## Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)

- How much peak memory does running AdamW require?

num_params = vocab_size  ***d_m + num_layers*  * (2*d_m + 4 * d_m *d_m + 3 * d_ff * d_m) + d_m

**parameters_memory** = num_params * 4                                 # 1.64B * 4 for our gpt2 xl

**gradients**: each parameters get updated/backpropped so, each have gradient, so, memory_gradients =              num_params * 4

**optimizer state**: m and v for each parameter = 2* num_params * 4bytes

**activations:**

| Component                 | Shape              | Elements       |
| ------------------------- | ------------------ | -------------- |
| RMSNorm(s) — 2 per block | `(B, L, d)` each | `2·B·L·d` |
| Q, K, V projections       | `(B, L, d)` each | `3·B·L·d` |
| QKᵀ scores               | `(B, h, L, L)`   | `B·h·L²`  |
| softmax                   | `(B, h, L, L)`   | `B·h·L²`  |
| weighted sum of values    | `(B, L, d)`      | `B·L·d`    |
| output projection         | `(B, L, d)`      | `B·L·d`    |
| FFN W1 (up-proj)          | `(B, L, d_ff)`   | `B·L·d_ff` |
| FFN W3                    | `(B, L, d_ff)`   | `B·L·d_ff` |
| SiLU (gate branch)        | `(B, L, d_ff)`   | `B·L·d_ff` |
| elementwise product       | `(B, L, d_ff)`   | `B·L·d_ff` |
| FFN W2 (down-proj)        | `(B, L, d)`      | `B·L·d`    |

**Outside the blocks (once):**

| Component                         | Shape             | Elements        |
| --------------------------------- | ----------------- | --------------- |
| final RMSNorm                     | `(B, L, d)`     | `B·L·d`     |
| output embedding (logits)         | `(B, L, vocab)` | `B·L·vocab` |
| cross-entropy (softmax on logits) | `(B, L, vocab)` | `B·L·vocab` |

- b) with gpt2 xl, What is the maximum batch size you can use and still fit within 80GB memory?

a =num_layers * (8*L *d + 2* *L^2  * h + 4*L*dff) + L*dm + 2*L*vocabsize
b = 16*num_params

total_memory = a*B + b = 4.1B * B * 4bytes + 16 * 1.64

for 80B memory, **B can be approx at most = 3**

- c) How many FLOPs does running one step of AdamW take?

num_params = 1.64B

about ~20 · num_params

d) number of hours training would take for H100:

forward = 3.52 * 10^12

backward = 2*forward

one training: 10.5T

totaltime = 10.5 * batchsize * numsteps = **200 days**
