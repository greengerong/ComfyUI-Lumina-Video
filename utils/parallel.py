import datetime
import os
import socket
import subprocess
from time import sleep
from typing import Any, Tuple

import torch
import torch.distributed as dist


def find_free_port(start_port: int, end_port: int):
    """
    Find a free port within the specified range.
    """
    for port in range(start_port, end_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))  # Try to bind to the port
            s.close()  # Close the socket if successful
            return port
        except OSError as e:
            # print(f"Port {port} is in use, trying next port.")
            continue
    raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")


def _setup_dist_env_from_slurm(args):
    while not os.environ.get("MASTER_ADDR", ""):
        os.environ["MASTER_ADDR"] = (
            subprocess.check_output(
                "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" % os.environ["SLURM_NODELIST"],
                shell=True,
            )
            .decode()
            .strip()
        )
        sleep(1)
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]


_INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP = None, None
_LOCAL_RANK, _LOCAL_WORLD_SIZE = -1, -1


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_local_world_size() -> int:
    return _LOCAL_WORLD_SIZE


def distributed_init(args):
    if any([x not in os.environ for x in ["RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]]):
        _setup_dist_env_from_slurm(args)

    dist.init_process_group("nccl", init_method="env://", timeout=datetime.timedelta(seconds=2 * 60 * 60))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    global _LOCAL_RANK, _LOCAL_WORLD_SIZE
    _LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    _LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    global _INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP
    local_ranks, local_world_sizes = [
        torch.empty([dist.get_world_size()], dtype=torch.long, device="cuda") for _ in (0, 1)
    ]
    dist.all_gather_into_tensor(local_ranks, torch.tensor(get_local_rank(), device="cuda"))
    dist.all_gather_into_tensor(local_world_sizes, torch.tensor(get_local_world_size(), device="cuda"))
    local_ranks, local_world_sizes = local_ranks.tolist(), local_world_sizes.tolist()
    node_ranks = [[0]]
    for i in range(1, dist.get_world_size()):
        if len(node_ranks[-1]) == local_world_sizes[i - 1]:
            node_ranks.append([])
        else:
            assert local_world_sizes[i] == local_world_sizes[i - 1]
        node_ranks[-1].append(i)
    for ranks in node_ranks:
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            assert _INTRA_NODE_PROCESS_GROUP is None
            _INTRA_NODE_PROCESS_GROUP = group
    assert _INTRA_NODE_PROCESS_GROUP is not None

    if min(local_world_sizes) == max(local_world_sizes):
        for i in range(get_local_world_size()):
            group = dist.new_group(list(range(i, dist.get_world_size(), get_local_world_size())))
            if i == get_local_rank():
                assert _INTER_NODE_PROCESS_GROUP is None
                _INTER_NODE_PROCESS_GROUP = group
        assert _INTER_NODE_PROCESS_GROUP is not None


def get_intra_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, "Intra-node process group is not initialized."
    return _INTRA_NODE_PROCESS_GROUP


def get_inter_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, "Intra- and inter-node process groups are not initialized."
    return _INTER_NODE_PROCESS_GROUP


class _AllToAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any) -> torch.Tensor:
        assert gather_dim != scatter_dim
        assert 0 <= gather_dim < input_.ndim
        assert 0 <= scatter_dim < input_.ndim
        world_size = dist.get_world_size(group)
        assert input_.size(scatter_dim) % world_size == 0

        ctx.gather_dim = gather_dim
        ctx.scatter_dim = scatter_dim
        ctx.group = group

        if world_size == 1:
            return input_

        inputs = [x.contiguous() for x in input_.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(x) for x in inputs]
        dist.all_to_all(outputs, inputs, group=group)

        return torch.cat(outputs, dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        group = ctx.group
        world_size = dist.get_world_size(group)
        gather_dim = ctx.gather_dim
        scatter_dim = ctx.scatter_dim

        if world_size == 1:
            return grad_output, None, None, None

        grad_outputs = [x.contiguous() for x in grad_output.chunk(world_size, dim=gather_dim)]
        grad_inputs = [torch.empty_like(x) for x in grad_outputs]

        dist.all_to_all(grad_inputs, grad_outputs, group=group)

        return torch.cat(grad_inputs, dim=scatter_dim), None, None, None


def all_to_all(
    input_: torch.Tensor,
    gather_dim: int,
    scatter_dim: int,
    gather_pad: int = 0,
    scatter_pad: int = 0,
    group: Any = None,
) -> torch.Tensor:

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if gather_pad > 0:
        if rank == world_size - 1:
            gather_pad_shape = list(input_.shape)
            gather_pad_shape[gather_dim] = gather_pad
            t_gather_pad = torch.zeros(gather_pad_shape, dtype=input_.dtype, device=input_.device)
            input_ = torch.cat([input_, t_gather_pad], dim=gather_dim)

    if scatter_pad > 0:
        scatter_pad_shape = list(input_.shape)
        scatter_pad_shape[scatter_dim] = scatter_pad
        t_scatter_pad = torch.zeros(scatter_pad_shape, dtype=input_.dtype, device=input_.device)
        input_ = torch.cat([input_, t_scatter_pad], dim=scatter_dim)

    result = _AllToAllFunction.apply(input_, gather_dim, scatter_dim, group)

    if gather_pad > 0:
        result = torch.narrow(result, gather_dim, 0, result.size(gather_dim) - gather_pad)

    if scatter_pad > 0:
        if rank == world_size - 1:
            result = torch.narrow(result, scatter_dim, 0, result.size(scatter_dim) - scatter_pad)

    return result


_SEQUENCE_PARALLEL_GROUP = None

_NEIGHBOR_GROUP_CACHE = {}


def set_sequence_parallel(sp_size: int) -> None:
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0

    global _SEQUENCE_PARALLEL_GROUP

    if sp_size in _NEIGHBOR_GROUP_CACHE:
        _SEQUENCE_PARALLEL_GROUP = _NEIGHBOR_GROUP_CACHE[sp_size]

    else:
        for j in range(world_size // sp_size):
            group_ranks = [(j * sp_size + k) for k in range(sp_size)]
            group = dist.new_group(ranks=group_ranks)
            if dist.get_rank() in group_ranks:
                _NEIGHBOR_GROUP_CACHE[sp_size] = group
                _SEQUENCE_PARALLEL_GROUP = group

    assert _SEQUENCE_PARALLEL_GROUP is not None


def get_sequence_parallel_group():
    assert _SEQUENCE_PARALLEL_GROUP is not None, "Sequence parallelism is not initialized."
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_rank():
    return dist.get_rank(get_sequence_parallel_group())


def get_sequence_parallel_world_size():
    return dist.get_world_size(get_sequence_parallel_group())


def get_sequence_parallel_src_rank():
    rank = dist.get_rank()
    sp_size = get_sequence_parallel_world_size()
    src_rank = rank - rank % sp_size
    return src_rank


def _sp_split(input_: torch.Tensor) -> torch.Tensor:
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if sp_size == 1:
        return input_
    assert input_.size(0) % sp_size == 0
    return input_.chunk(sp_size, dim=0)[sp_rank].contiguous()


def _sp_scatter(input_: torch.Tensor) -> torch.Tensor:
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    sp_src = get_sequence_parallel_src_rank()
    sp_group = get_sequence_parallel_group()
    if sp_size == 1:
        return input_
    assert input_.size(0) % sp_size == 0
    output = torch.empty(
        [x if i != 0 else x // sp_size for i, x in enumerate(input_.size())], dtype=input_.dtype, device=input_.device
    )
    dist.scatter(
        output,
        [x.contiguous() for x in input_.chunk(sp_size, dim=0)] if sp_rank == 0 else None,
        src=sp_src,
        group=sp_group,
    )
    return output


def _sp_gather(input_: torch.Tensor) -> torch.Tensor:
    sp_group = get_sequence_parallel_group()
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if sp_size == 1:
        return input_
    output = [torch.empty_like(input_) for _ in range(sp_size)]
    dist.all_gather(output, input_, group=sp_group)
    return torch.cat(output, dim=0)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        if rank0_only:
            return _sp_scatter(input_)
        else:
            return _sp_split(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return _sp_gather(grad_output / get_sequence_parallel_world_size()), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, rank0_only: bool = True) -> torch.Tensor:
        ctx.rank0_only = rank0_only
        return _sp_gather(input_)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        if ctx.rank0_only:
            return _sp_scatter(grad_output) * get_sequence_parallel_world_size(), None
        else:
            return _sp_split(grad_output) * get_sequence_parallel_world_size(), None


def scatter_to_sequence_parallel_region(input_: torch.Tensor, rank0_only: bool = True) -> Tuple[torch.Tensor, int]:
    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()

    # pad_num = (sp_world_size - input_.shape[0] % sp_world_size) % sp_world_size
    pad_num = -input_.shape[0] % sp_world_size

    if pad_num > 0:
        input_ = torch.cat(
            [input_, torch.zeros([pad_num, *input_.shape[1:]], dtype=input_.dtype, device=input_.device)], dim=0
        )
        padded_len = input_.shape[0]
        assert padded_len // sp_world_size > pad_num, "only last sp rank expected to have padded tokens"

    result = _ScatterToSequenceParallelRegion.apply(input_, rank0_only)

    if pad_num > 0:
        if sp_rank == sp_world_size - 1:
            result = result[:-pad_num]

    return result, pad_num


def gather_from_sequence_parallel_region(
    input_: torch.Tensor, rank0_only: bool = True, pad_num: int = 0
) -> torch.Tensor:

    if pad_num > 0:
        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()
        if sp_rank == sp_world_size - 1:
            input_ = torch.cat(
                [input_, torch.zeros([pad_num, *input_.shape[1:]], dtype=input_.dtype, device=input_.device)], dim=0
            )

    result = _GatherFromSequenceParallelRegion.apply(input_, rank0_only)

    if pad_num > 0:
        result = result[:-pad_num]

    return result
