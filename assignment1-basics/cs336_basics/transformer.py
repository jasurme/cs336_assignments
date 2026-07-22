
%uv pip install einops
import torch
import einops
import math
import numpy as np

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device : torch.device = None, dtype:torch.dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.device=device
        self.dtype=dtype
        self.std= math.sqrt(2 / (self.in_features + self.out_features))
        self.W = torch.nn.Parameter(torch.empty(self.out_features, self.in_features))
        torch.nn.init.trunc_normal_(self.W, std=self.std, a=-3* self.std, b = 3* self.std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einops.einsum(self.W, x, "out in, ... in -> ... out")
        return y


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None ): # 𝑑model
        super().__init__()
        self.num_embeddings= num_embeddings
        self.embedding_dim = embedding_dim
        self.device=device
        self.dtype=dtype
        self.embedding_matrix = torch.nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(torch.int64)
        return self.embedding_matrix[token_ids]



class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.device=device
        self.dtype=dtype
        self.g = torch.nn.Parameter(torch.ones(self.d_model,))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self.rms(x)
        result = x/rms*self.g
        return result.to(in_dtype) #uv run pytest -k test_rmsnorm

    def rms(self, a):
        return torch.sqrt(torch.mean(torch.square(a), dim=2, keepdim=True) + self.eps)


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2=Linear(self.d_ff, self.d_model)
        self.w3=Linear(self.d_model, self.d_ff, )

    def forward(self, x):
        #FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2(SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥)
        #SiLU(𝑥) = 𝑥 ⋅ 𝜎(𝑥)

        return self.w2(self.w1(x) * torch.sigmoid(self.w1(x)) * self.w3(x))




class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None ):
        super().__init__()
        self.theta = theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.device=device
        positions = torch.arange(self.max_seq_len)
        k = torch.arange(self.d_k//2)
        freqs = self.theta ** (-2*k/self.d_k)
        angle_table = einops.einsum(positions, freqs, "max_seq_len, d_2 -> max_seq_len d_2")
        angle_paired = einops.repeat(angle_table, "i k -> i (k 2)")
        sin_table = torch.sin(angle_paired)
        cos_table = torch.cos(angle_paired)
        self.register_buffer("sintable_cache", sin_table, persistent=False)
        self.register_buffer("costable_cache", cos_table, persistent=False)

    def rot(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return einops.rearrange([ -x2, x1 ], "pair ... k -> ... (k pair)")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: #x - (..., seq_len, d_k)
        token_positions = token_positions.to(torch.int64)
        sins = self.sintable_cache[token_positions] # still 2d
        conses = self.costable_cache[token_positions]

        return (x * conses) + (self.rot(x) * sins)



def softmax(tnsr, dim_i,temperature = 1):
    max_val = torch.max(tnsr, dim=dim_i, keepdim=True).values
    exp_tnsr=torch.exp((tnsr - max_val)/temperature)
    exp_sum = torch.sum(exp_tnsr, dim=dim_i, keepdim=True)
    return exp_tnsr/exp_sum



def scaled_dot_product_attention(Q, K, V, mask_tensor=None): # 𝑄𝐾⊤/√𝑑𝑘
    q_k = einops.einsum(Q, K, "batch_size ... seq_q d_k , batch_size ... seq_k d_k -> batch_size ... seq_q seq_k")
    inp = q_k/torch.sqrt(torch.tensor(K.shape[-1]))
    if mask_tensor is not None:
        inp = inp.masked_fill(~mask_tensor, float("-inf"))
    attention_weights = torch.softmax(inp, dim=-1)
    output = einops.einsum(
        attention_weights, V,
        "batch_size ... seq_q seq_k, batch_size ... seq_k d_v -> batch_size ... seq_q d_v"
    )

    return output





class CasualMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,d_model, num_heads, theta, max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads = num_heads
        self.d_v = self.d_model // self.num_heads
        self.d_k = self.d_v
        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k,max_seq_len)

    def forward(self, x,token_positions=None):
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) ==1


        Q = self.q_proj(x)
        Q = einops.rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        K = self.k_proj(x)
        K = einops.rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        V = self.v_proj(x)
        V = einops.rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=self.num_heads)

        # apply RoPE
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)


        scores = einops.einsum(Q, K, "... heads tokens_q d_q, ... heads tokens_k d_q -> ... heads tokens_q tokens_k")
        scores = scores.masked_fill(~mask,float("-inf"))
        after_softmax = torch.softmax(scores/math.sqrt(self.d_k), dim=-1)
        heads = einops.einsum(after_softmax, V, "... heads tokens_q tokens_k, ... heads tokens_k d_v -> ... heads tokens_q d_v")

        multihead = einops.rearrange(heads, "... heads seq d_v -> ... seq (heads d_v)")

        self_multihead = self.output_proj(multihead)
        return self_multihead





def transformer_block(d_model,num_heads, d_ff, max_seq_len, theta, weights:dict, in_features, idx):
    rmsnorm = RMSNorm(in_features)
    rmsnorm.load_state_dict({"g": weights[f'layers.{idx}.ln1.weight']})
    #ln1.weight
    in_features_rms = rmsnorm.forward(in_features)
    multihead = CasualMultiHeadSelfAttention(d_model, num_heads)
    selfattention = multihead.multihead_self_attention(
        weights[f'layers.{idx}.attn.q_proj.weight'],
        weights[f'layers.{idx}.attn.k_proj.weight'],
        weights[f'layers.{idx}.attn.v_proj.weight'],
        weights[f'layers.{idx}.attn.output_proj.weight'],
        in_features_rms,
        True,
        theta,
        max_seq_len,
        token_positions=None
    )
    selfattention_resnet = in_features + selfattention
     #ln2.weight
    rmsnorm2 = RMSNorm(d_model)
    rmsnorm2.load_state_dict({"g": weights[f'layers.{idx}.ln2.weight']})
    in_features_rms2 = rmsnorm2.forward(selfattention_resnet)

    ff = PositionwiseFeedForward(d_model, d_ff)
    ff.load_state_dict({"w1": weights[f'layers.{idx}.ffn.w1.weight'], "w2": weights[f'layers.{idx}.ffn.w2.weight'], "w3": weights[f'layers.{idx}.ffn.w3.weight']})
    ff_out = ff.forward(in_features_rms2)

    final = selfattention_resnet + ff_out
    return final


def transformer_lm(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, weights, in_indices):
    embedding = Embedding(vocab_size, d_model)
    embedding.load_state_dict({"embedding_matrix": weights['token_embeddings.weight']})
    embeddings = embedding.forward(in_indices)
    for i in range(num_layers):
        x = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, weights, embeddings, i)
        embeddings = x

    #rmsnorm after transformer blocks
    rmsnorm = RMSNorm(d_model)
    rmsnorm.load_state_dict({"g": weights['ln_final.weight']})
    rmsnorm_out = rmsnorm.forward(x)
    linear = Linear(d_model, vocab_size)
    linear.load_state_dict({"W": weights['lm_head.weight']})
    return linear.forward(rmsnorm_out)

class TransformerBlock(torch.nn.Module):
    def __init__(self,d_model,num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.rmsnorm1 = RMSNorm(self.d_model)
        self.attention = CasualMultiHeadSelfAttention(self.d_model, self.num_heads, self.theta, self.max_seq_len)
        self.rmsnorm2 = RMSNorm(self.d_model)
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_ff)

    def forward(self, x, token_positions=None):

        y = x + self.attention.forward(self.rmsnorm1.forward(x))

        return y + self.ffn.forward(self.rmsnorm2.forward(y))




class TransformerLM(torch.nn.Module):
    def __init__(self,vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    def forward(self, in_indices, token_positions=None):
        x = self.embedding.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x, token_positions)

        finalnormout = self.ln_final.forward(x)
        logits = self.lm_head.forward(finalnormout)
        return logits






def cross_entropy(inputs, targets):


    max_val= torch.max(inputs, dim=1, keepdim=True).values
    batch_indices = torch.arange(len(targets))
    target_logits = inputs[batch_indices, targets].unsqueeze(1)
    losses = -target_logits + max_val+ torch.log(torch.sum(torch.exp(inputs-max_val), dim=1, keepdim=True))
    return losses.mean()




class AdamW(torch.optim.Optimizer):
    def __init__(self,params, lr, betas=(0.9, 0.998), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    def step(self ,closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr , (b1, b2),eps,  weight_decay= group['lr'], group['betas'],group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                statee = self.state[p]
                t = statee.get("t", 0) + 1
                m, v = statee.get("m",torch.zeros_like(p) ), statee.get("v",torch.zeros_like(p) )
                m = b1* m + (1-b1)*grad
                v = b2*v + (1-b2)* grad**2
                p.data =p.data - lr*m*math.sqrt(1-b2**t)/((torch.sqrt(v) + eps) * (1-b1**t)) - lr*weight_decay*p.data
                statee['t'] =t
                statee['m'] = m
                statee['v'] = v
        return loss



def learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    else: #1 2 (1 + cos( 𝑡−𝑇𝑤 𝑇𝑐−𝑇𝑤 𝜋))(𝛼max − 𝛼min).
        lr = min_learning_rate + 1/2 * (1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*torch.pi)) * (max_learning_rate-min_learning_rate)

    return lr


def gradient_clipping(parameters,max_l2_norm):
    grads = [param.grad for param in parameters if param.grad is not None]
    if len(grads) ==0:
        return None
    norms = [g.norm(2) for g in grads]
    norms_tensor = torch.stack(norms)
    global_norm = torch.norm(norms_tensor, p=2)
    if global_norm > max_l2_norm:
        clip =  max_l2_norm / (global_norm + 1e-6)
        for g in grads:
            g.mul_(clip)


def data_loading(dataset,batch_size, context_length,device='mps'):
    starts = np.random.randint(0, len(dataset)-context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)[None, :]
    inputs = dataset[idx]
    inputs = torch.tensor(inputs, device=device, dtype=torch.long)
    targets = dataset[idx+1]
    targets = torch.tensor(targets, device=device, dtype=torch.long)

    return inputs,targets


def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    allstate = {"model_state": model_state,
                "optimizer_state": optimizer_state,
                "iteration_no": iteration
                }
    torch.save(allstate, out)

def load_checkpoint(src, model, optimizer):
    allstate = torch.load(src)
    model.load_state_dict(allstate['model_state'])
    optimizer.load_state_dict(allstate['optimizer_state'])
    return allstate['iteration_no']


# context_length=400000

def decoding(model, prompt_tokens, top_p, temperature=1, max_tokens=10000):

    end_tokenid = 50256 # '<|endoftext|>'

    for i in range(max_tokens):

        logits = model.predict(prompt_tokens)
        q = softmax(logits, 0, temperature)
        q_sorted, indexessorted = torch.sort(q, descending=True)
        cum_probs = torch.cumsum(q_sorted, dim=0)

        top_k =torch.searchsorted(cum_probs, top_p).item() # number like 2 or 3

        top_probs = q_sorted[:top_k+1] # e.g. 3 top probabilities like [0.7,0.2,0.1]
        top_indices = indexessorted[:top_k+1]
        new_probs_dist = top_probs / top_probs.sum()
        sampled_idx = torch.multinomial(new_probs_dist, num_samples=1).item()
        next_token = top_indices[sampled_idx].item()

        if next_token == end_tokenid:
            break

        prompt_tokens.append(next_token)
        prompt_tokens = prompt_tokens[:context_length]

    return prompt_tokens
