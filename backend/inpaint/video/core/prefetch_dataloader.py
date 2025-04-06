import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader


class PrefetchGenerator(threading.Thread):
    """일반적인 프리페치 제너레이터.
 
    참조:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
 
    Args:
        generator: 파이썬 제너레이터.
        num_prefetch_queue (int): 프리페치 큐의 수.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """데이터 로더의 프리페치 버전.
 
    참조:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#
 
    TODO:
    단일 GPU 및 ddp(다중 GPU)에서 테스트해야 합니다. ddp에는 알려진 문제가 있습니다.
 
    Args:
        num_prefetch_queue (int): 프리페치 큐의 수.
        kwargs (dict): 데이터 로더의 다른 인수.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU 프리페처.
 
    Args:
        loader: 데이터 로더.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA 프리페처.
 
    참조:
    https://github.com/NVIDIA/apex/issues/304#
 
    더 많은 GPU 메모리를 소비할 수 있습니다.
 
    Args:
        loader: 데이터 로더.
        opt (dict): 옵션.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch는 사전입니다.
        except StopIteration:
            self.batch = None
            return None
        # 텐서를 GPU로 전송
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
