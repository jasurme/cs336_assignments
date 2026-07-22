import torch
import math
class FA2_Forward_PyTorch():
    def __init__(self, B_r=16, B_c=16):
        self.B_r= B_r
        self.B_c= B_c

    def flashattention2_forward_pytorch(self, Q, K, V, is_causal): # Q,K,V - (N,d)
        B_q, N_q , d = Q.shape
        B_k, N_k, d = K.shape   # maybe in practice, usuallybatchsizes are equal? B_k = B_q ?
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
            Oi = torch.zeros_like(blocksO[i])
            li = torch.zeros(B_q, self.B_r, 1)
            mi = torch.ones(B_q, self.B_r, 1) * float('-inf')


            for j in range(Tc):
                Kj = blocksK[j] # (3,128)    #(B_c, d)
                Vj = blocksV[j]      #(B_c, d)
                Sij = (Qi @ Kj.T) / torch.sqrt(d)     #, (B_r, B_c)
                if is_causal:
                    
                mij1 = mi  #2d (B_r,1), same as mi[j]
                mi = torch.max(mij1, torch.max(Sij, dim=1, keepdim=True).values) #2d (B_r,1)
                Pij = torch.exp(Sij - mi) #(B_r, B_c)
                correction_factor = torch.exp(mij1 - mi) # (B_r, 1)
                li = correction_factor * li+ torch.sum(Pij, dim=-1, keepdim=True)#  (B_r, 1) @ (B_r, 1)      # rowsum(Pij) is (B_r,1)
                Oi = correction_factor * Oi  + Pij @ Vj # (B_r, B_r) @


            blocksO[i] = torch.pow(li, -1) * Oi
            blocksL[i] = mi + torch.log(li)

        O = torch.cat(blocksO, dim=-1)
        L = torch.cat(blocksL, dim=-1)

        return O, L



class FA2FunctionPytorchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V, is_causal=False):
        clss = FA2_Forward_PyTorch(32,32)
        O, L = clss.flashattention2_forward_pytorch(Q, K, V, is_causal)

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx):
        raise NotImplementedError('not implemented')
