import os
import torch


def get_world_size():
    """mpi 함수를 호출하지 않고 OMPI 월드 크기를 찾습니다.
    :rtype: int
    """
    if os.environ.get('PMI_SIZE') is not None:
        return int(os.environ.get('PMI_SIZE') or 1)
    elif os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
    else:
        return torch.cuda.device_count()


def get_global_rank():
    """mpi 함수를 호출하지 않고 OMPI 월드 랭크를 찾습니다.
    :rtype: int
    """
    if os.environ.get('PMI_RANK') is not None:
        return int(os.environ.get('PMI_RANK') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    else:
        return 0


def get_local_rank():
    """mpi 함수를 호출하지 않고 OMPI 로컬 랭크를 찾습니다.
    :rtype: int
    """
    if os.environ.get('MPI_LOCALRANKID') is not None:
        return int(os.environ.get('MPI_LOCALRANKID') or 0)
    elif os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') is not None:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)
    else:
        return 0


def get_master_ip():
    if os.environ.get('AZ_BATCH_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCH_MASTER_NODE').split(':')[0]
    elif os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE') is not None:
        return os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE')
    else:
        return "127.0.0.1"
