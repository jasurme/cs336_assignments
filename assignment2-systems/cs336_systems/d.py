import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import statistics
import copy
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
def distributed_demo(rank, world_size):
    setup(rank, world_size)
    # n_elements = [262144, 2,621,440, 26,214,400, 268435456]

    data = torch.randint(0, 1, (268435456,), dtype=torch.float32)
    # print(f"rank {rank} data (before all-reduce): {data}")
    # warm up:
    for i in range(5): # warm up steps
        dist.all_reduce(data, async_op=False)

    dist.barrier()
    n_steps = 100
    local_timings = []
    for i in range(n_steps):
        start = timeit.default_timer()
        # run more, and take mean
        dist.all_reduce(data, async_op=False)
        end = timeit.default_timer()
        local_timings.append(end-start)
    dist.barrier()
    global_timings = [None for _ in range(world_size)]
    local_data = local_timings
    dist.all_gather_object(global_timings, local_data)
    if rank == 0:
        mean_values_for_num_steps = [sum(global_timings[r][i] for r in range(world_size)) / world_size for i in range(n_steps)]

        final_mean = statistics.mean(mean_values_for_num_steps)

        print(f"100MB world_size {world_size} taken: {final_mean}")




    dist.destroy_process_group()




    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 6

    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)



class Naive_DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)


    def forward(self, input_data):
        output = self.module(input_data)
        return output

    def gradient_synchr(self):

        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad /= dist.get_world_size()

    # flattened gradient strategy
    def gradient_synchr_flat(self):
        grad_list = [param.grad for param in self.module.parameters() if param.grad is not None]
        flattened_grads = torch._utils._flatten_dense_tensors(grad_list)
        flattened_grads /= dist.get_world_size()

        dist.all_reduce(flattened_grads)

        unflattered_tensors = torch._utils._unflatten_dense_tensors(flattened_grads, grad_list)

        for org_grad,new_grad in zip(grad_list,unflattered_tensors):
            org_grad.copy_(new_grad)

    # overlapping computation of backward pass with communication of gradients



class Overlapping_DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        self.handles = []
        def post_hook(param):
            param.grad /= dist.get_world_size()
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append(handle)


        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(post_hook)



    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *inputs, **kwargs):
            output = self.module(*inputs, **kwargs)
            return output


# optimizer state sharding

class Optimizer_State_Sharder(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.all_params = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owners = []
        self.local_params = []
        self.global_param_counter = 0


        super().__init__(params, kwargs)
        self.local_optimizer = optimizer_cls(self.local_params, **kwargs)

    def step(self, closure=None, **kwargs):
        result = self.local_optimizer.step(closure=closure, **kwargs)
        with torch.no_grad():
            for param, owner in zip(self.all_params, self.owners):
                dist.broadcast(param, src=owner)

        return result




    def add_param_group(self, param_group):
        param_group1 = param_group.copy()
        param_group1['params'] = list(param_group1['params'])
        super().add_param_group(param_group1)


        for param in param_group1['params']:
            owner = self.global_param_counter % self.world_size
            self.owners.append(owner)
            self.all_params.append(param)
            if owner == self.rank:
                self.local_params.append(param)

            self.global_param_counter+=1


import math
from cs336_basics.model import Linear, Embedding
class FullySharded_DataParallelTraining(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.sharded_layers = {}
        self.sharded_metadata = {}
        self.module_to_name_mapping = {}
        self.param_to_name_metadata_mapping = {}
        self.replicated_params = []
        self.handles = []
        self.schedule_allgathers = []
        with torch.no_grad():
            for name, modle in self.module.named_modules():
                if isinstance(modle, Linear) or isinstance(modle, Embedding):
                    self.sharded_layers[name] = modle
                    assert modle.weight.numel() % self.world_size == 0
                    shard_numel = modle.weight.numel() // self.world_size
                    self.sharded_metadata[name] = {"module": modle, "parameter": modle.weight, "original_shape": modle.weight.shape, "n_elements": modle.weight.numel(), "dtype": modle.weight.dtype, "shard_numel": shard_numel}
                    self.module_to_name_mapping[modle] = name
                    flattened = modle.weight.data.view(-1)

                    self.param_to_name_metadata_mapping[modle.weight] = name

                    rank_slice = flattened[self.rank * shard_numel: self.rank * shard_numel + shard_numel].clone()
                    modle.weight.data = rank_slice
                    self.sharded_metadata[name]['local_shard'] = rank_slice

                    # add forward prehook + posthook
                    modle.register_forward_pre_hook(self.forward_pre_hook)
                    modle.register_forward_hook(self.forward_post_hook)
                    #register backward prehook

                    modle.register_full_backward_pre_hook(self.backward_pre_hook)

                    modle.weight.register_post_accumulate_grad_hook(self.backward_post_hook)

                    self.schedule_allgathers.append(modle)


        for param in self.module.parameters():
            if param not in self.param_to_name_metadata_mapping:
                self.replicated_params.append(param)

    def backward_post_hook(self, parameter):
        local_metadata = self.sharded_metadata[self.param_to_name_metadata_mapping[parameter]]

        flattened = parameter.grad.detach().reshape(-1)
        flattened = flattened.to(local_metadata['local_shard'].dtype)
        red_scat_slice = torch.empty(local_metadata['shard_numel'], dtype=flattened.dtype, device=flattened.device)
        handle = dist.reduce_scatter_tensor(input=flattened,
                        output=red_scat_slice, async_op=True)
        self.handles.append({
            "handle": handle,
            "parameter": parameter,
            "output_grad_shard": red_scat_slice,
            "flattened": flattened

        })



        # red_scat_slice /= self.world_size

        parameter.grad = None
        parameter.data = local_metadata['local_shard']
        # parameter.grad = red_scat_slice
        del local_metadata['full_matrix']


    def backward_pre_hook(self, module, grad_output):
        layer_metadata = self.sharded_metadata[self.module_to_name_mapping[module]]
        full_matrix = self.reconstruct(layer_metadata)
        layer_metadata['full_matrix'] = full_matrix
        module.weight.data = full_matrix

    def forward_pre_hook(self, module, input):
        # before doing forward calc, we need to retrieve full matrix for fsdp
        fullmatrix = self.reconstruct(self.sharded_metadata[self.module_to_name_mapping[module]])
        self.sharded_metadata[self.module_to_name_mapping[module]]['fullmatrix'] = fullmatrix
        module.weight.data = fullmatrix


    def forward_post_hook(self, module, inputs, output):
        metadata = self.sharded_metadata[self.module_to_name_mapping[module]]

        module.weight.data = metadata['local_shard']
        del metadata['fullmatrix']


    def reconstruct(self, layer_metadata, for_compute=True):

        detached = layer_metadata['parameter'].detach()
        if self.compute_dtype is not None and for_compute:
            detached =detached.to(self.compute_dtype)

        fullmatrix_flattened = torch.empty(layer_metadata['n_elements'], device=layer_metadata['parameter'].device, dtype=detached.dtype)

        handle = dist.all_gather_into_tensor(fullmatrix_flattened, detached, async_op=True)

        fullmatrix = fullmatrix_flattened.view(layer_metadata['original_shape'])
        return fullmatrix

    def fsdp_gather_full_params(self):
        gathered_params = {}
        for name, param in self.module.named_parameters():

            if param in self.param_to_name_metadata_mapping:
                full_tensor = self.reconstruct(self.sharded_metadata[self.param_to_name_metadata_mapping[param]], for_compute=False)
                full_tensor = full_tensor.to(param.dtype)
                gathered_params[name] = full_tensor
            else:
                gathered_params[name] = param.data.detach().clone()

        return gathered_params

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle_dict in self.handles:
            handle_dict['handle'].wait()
            handle_dict['output_grad_shard'] /= self.world_size
            handle_dict['parameter'].grad = handle_dict['output_grad_shard']
        self.handles.clear()
        for param in self.replicated_params:
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad /= self.world_size
