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



class FA2FunctionTritonAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx):
        pass

    @staticmethod
    def backward(ctx):
        return NotImplementedError('not implemented')


# ---------    ---------------------------------------------
# triton matmul blocked
# ---------    --------------------------------------------

import triton
import torch
import triton.language as tl

def triton_autotune_config():
  return [
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
  ]

@triton.autotune(configs= triton_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(X_ptr, Y_ptr, Z_ptr,
                  M, K, N,
                  stride_xm, stride_xk,
                  stride_yk, stride_yn,
                  stride_zm, stride_zn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
  pid = tl.program_id(axis=0)
  num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

  block_m = pid // num_block_cols
  block_n = pid % num_block_cols

  offs_m_rows = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_n_cols = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

  acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Define offs_k inside the loop to represent the current K block
    offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Pointers for X (A) - recompute in each iteration
    a_ptrs = X_ptr + (offs_m_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Mask for loading X (A) - check against total K
    a_mask = (offs_m_rows[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Pointers for Y (B) - recompute in each iteration
    b_ptrs = Y_ptr + (offs_k[:, None] * stride_yk + offs_n_cols[None, :] * stride_yn)
    # Mask for loading Y (B) - check against total K
    b_mask = (offs_k[:, None] < K) & (offs_n_cols[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc += tl.dot(a, b)

  z = acc.to(X_ptr.dtype.element_ty)

  offs_c = stride_zm * offs_m_rows[:,None] + stride_zn * offs_n_cols[None, :]

  tl.store(Z_ptr+ offs_c, z, mask= (offs_m_rows[:,None] < M) & (offs_n_cols[None, :] < N))

def matmul(x, y):
  M, K = x.shape
  K, N = y.shape
  z = torch.empty((M, N), device='cuda')

  grid=lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
  matmul_kernel[grid](
      x,y,z,
      M, K, N,
      x.stride(0), x.stride(1
      y.stride(0), y.stride(1),
      z.stride(0), z.stride(1)
  )
  return z


X = torch.randn((1024, 2048), device='cuda')
Y = torch.randn((2048, 32), device='cuda')

Z = matmul(X, Y)

Z - X @ Y



# ---------    ---------------------------------------------
# triton matmul **grouped**
# ---------    --------------------------------------------



import triton
import torch
import triton.language as tl

def triton_autotune_config_grouped():
  return [
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
      triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
  ]

@triton.autotune(configs= triton_autotune_config_grouped(), key=['M', 'N', 'K'])
@triton.jit
def grouped_matmul_kernel(X_ptr, Y_ptr, Z_ptr,
                  M, K, N,
                  stride_xm, stride_xk,
                  stride_yk, stride_yn,
                  stride_zm, stride_zn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
  pid = tl.program_id(axis=0)

  num_pids_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pids_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pids_in_group = GROUP_SIZE_M * num_pids_n # variable name is a bit confusing here.
  # in below case, GROUP_SIZE_M = 3, num_pids_n = 6, so num_pids_in_group = 3*6 = 18. if you look below. number of pids in one group is 9(3x3 matrix),
  # but num_pids_in_group is refering to how many pids it takes to go to next row of group.
  # if you look at this big matrix as matrix of groups, it is just (3x2) matrix, so, to go from (0,0) to (1,0), it takes num_pids_in_group

  #       0   1   2  |  3   4   5
  #    +-------------+-------------
  #  0 |  0   3   6  |  9  12  15
  #  1 |  1   4   7  | 10  13  16
  #  2 |  2   5   8  | 11  14  17
  #    |-------------+-------------
  #  3 | 18  21  24  | 27  30  33
  #  4 | 19  22  25  | 28  31  34
  #  5 | 20  23  26  | 29  32  35
  #    |-------------+-------------
  #  6 | 36  39  42  | 45  48  51
  #  7 | 37  40  43  | 46  49  52
  #  8 | 38  41  44  | 47  50  53

  group_id = pid // num_pids_in_group # 11 // 18 = 0
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pids_m- first_pid_m,GROUP_SIZE_M)
  where_in_group = pid % num_pids_in_group #
  which_row = where_in_group % group_size_m # for examlpe, in 2nd row group, it repeats as, which_row = 0,1,2,   0,1,2

  pid_m = first_pid_m + which_row # now it repeats as    3,4,5,   3,4,5.  3,4,5
  pid_n = where_in_group // group_size_m # e.g. 32 (where_in_grou = 32 % 18= 14 ) -> 14 // 3 = 4?
  # first_pid_m = 1*3=3   , which_row = 14 % 3 = 2, pid_m= 3 +2 = 5 (! correct), pid_n = 14 // 3 = 4




  num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

  block_m = pid // num_block_cols
  block_n = pid % num_block_cols

  offs_m_rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # change block_m -> pid_m
  offs_n_cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # change block_n -> pid_n

  acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Define offs_k inside the loop to represent the current K block
    offs_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Pointers for X (A) - recompute in each iteration
    a_ptrs = X_ptr + (offs_m_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Mask for loading X (A) - check against total K
    a_mask = (offs_m_rows[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Pointers for Y (B) - recompute in each iteration
    b_ptrs = Y_ptr + (offs_k[:, None] * stride_yk + offs_n_cols[None, :] * stride_yn)
    # Mask for loading Y (B) - check against total K
    b_mask = (offs_k[:, None] < K) & (offs_n_cols[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc += tl.dot(a, b)

  z = acc.to(X_ptr.dtype.element_ty)

  offs_c = stride_zm * offs_m_rows[:,None] + stride_zn * offs_n_cols[None, :]

  tl.store(Z_ptr+ offs_c, z, mask= (offs_m_rows[:,None] < M) & (offs_n_cols[None, :] < N))

def matmul(x, y):
  M, K = x.shape
  K, N = y.shape
  z = torch.empty((M, N), device='cuda')

  grid=lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
  matmul_kernel[grid](
      x,y,z,
      M, K, N,
      x.stride(0), x.stride(1),
      y.stride(0), y.stride(1),
      z.stride(0), z.stride(1)
  )
  return z


X = torch.randn((1024, 16384), device='cuda')
Y = torch.randn((16384, 32), device='cuda')

Z = matmul(X, Y)

torch.allclose(Z, X @ Y, atol=1e-2, rtol=0), Z - X @ Y
