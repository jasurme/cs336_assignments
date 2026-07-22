import torch
import math
class FA2_Forward_PyTorch():
    def __init__(self, B_r=16, B_c=16):
        self.B_r= B_r
        self.B_c= B_c

    def flashattention2_forward_pytorch(self, Q, K, V, is_causal): # Q,K,V - (N,d)
        B_q, N_q , d = Q.shape
        B_k, N_k, d_k = K.shape   # maybe in practice, usuallybatchsizes are equal? B_k = B_q ?
        O = torch.empty_like(Q)
        L = torch.zeros(B_q, N_q)

        blocksQ = torch.split(Q, self.B_r, dim=1) # tuple of blocks with size (B_r, d)
        blocksK = torch.split(K, self.B_c, dim=1)
        blocksV = torch.split(V, self.B_c, dim=1)
        blocksO = torch.split(O, self.B_r, dim=1)
        blocksL = torch.split(L, self.B_r, dim=-1)
        Tr = N_q // self.B_r
        Tc =  N_k // self.B_c



        for i in range(Tr):
            Qi = blocksQ[i] #  (B_r, d) e.g. (2,128)
            Oi = torch.zeros(B_q, self.B_r, d_k)
            li = torch.zeros(B_q, self.B_r, 1)
            mi = torch.ones(B_q, self.B_r, 1) * float('-inf')

            q_start = i * self.B_r
            q_end = q_start + self.B_r

            for j in range(Tc):
                Kj = blocksK[j] # (3,128)    #(B_c, d)
                Vj = blocksV[j]      #(B_c, d)
                Sij = (Qi @ Kj.transpose(-2, -1)) / torch.sqrt(torch.tensor(d))     #, (B_r, B_c)
                k_start = j * self.B_c
                k_end = k_start + self.B_c
                if is_causal:
                    q_positions = torch.arange(q_start, q_end)
                    k_positions = torch.arange(k_start, k_end)
                    causal_offset = N_k - N_q
                    causal_mask = (k_positions[None, :] > q_positions[:, None] + causal_offset)
                    Sij = Sij.masked_fill(causal_mask[None, :, :],  float("-inf"))

                mij1 = mi  #2d (B_r,1), same as mi[j]
                mi = torch.max(mij1, torch.max(Sij, dim=-1, keepdim=True).values) #2d (B_r,1)
                Pij = torch.exp(Sij - mi) #(B_r, B_c)
                correction_factor = torch.exp(mij1 - mi) # (B_r, 1)
                li = correction_factor * li+ torch.sum(Pij, dim=-1, keepdim=True)#  (B_r, 1) * (B_r, 1)      # rowsum(Pij) is (B_r,1)
                Oi = correction_factor * Oi  + Pij @ Vj # (B_r, d) + (B_r, B_c) @ (B_c, d) =  (B_r, d) +  (B_r, d)

            print('after j, inside i ', i)
            blocksO[i].copy_(Oi / li)
            blocksL[i].copy_((mi + torch.log(li)).squeeze())

        O = torch.cat(blocksO, dim=1)
        L = torch.cat(blocksL, dim=-1)

        return O, L



class FA2FunctionPytorchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V, is_causal=False):
        print('size Q', Q.shape)
        print('size V', V.shape)
        clss = FA2_Forward_PyTorch(32,32)
        O, L = clss.flashattention2_forward_pytorch(Q, K, V, is_causal)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx):
        raise NotImplementedError('not implemented')
