"""
    BasicSR의 LR 스케줄러 https://github.com/xinntao/BasicSR
"""
import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ 재시작 기능이 있는 MultiStep 학습률 스케줄링 방식.
    Args:
        optimizer (torch.nn.optimizer): Torch 옵티마이저.
        milestones (list): 학습률을 감소시킬 반복 횟수 목록.
        gamma (float): 감소 비율. 기본값: 0.1.
        restarts (list): 재시작 반복 횟수 목록. 기본값: [0].
        restart_weights (list): 각 재시작 반복에서의 재시작 가중치 목록.
            기본값: [1].
        last_epoch (int): _LRScheduler에서 사용됨. 기본값: -1.
    """
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=(0, ),
                 restart_weights=(1, ),
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """주기 목록에서 위치를 가져옵니다.
    주기 목록에서 오른쪽에 가장 가까운 숫자의 인덱스를 반환합니다.
    예: cumulative_period = [100, 200, 300, 400]
    iteration == 50이면 0 반환;
    iteration == 210이면 2 반환;
    iteration == 300이면 2 반환.
    Args:
        iteration (int): 현재 반복 횟수.
        cumulative_period (list[int]): 누적 주기 목록.
    Returns:
        int: 주기 목록에서 오른쪽에 가장 가까운 숫자의 위치.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ 재시작 기능이 있는 코사인 어닐링 학습률 스케줄링 방식.
    설정 예시:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    각각 10번의 반복을 갖는 4개의 사이클이 있습니다. 10번째, 20번째, 30번째 반복에서
    스케줄러는 restart_weights의 가중치로 재시작합니다.
    Args:
        optimizer (torch.nn.optimizer): Torch 옵티마이저.
        periods (list): 각 코사인 어닐링 사이클의 주기 목록.
        restart_weights (list): 각 재시작 반복에서의 재시작 가중치 목록.
            기본값: [1].
        eta_min (float): 최소 학습률. 기본값: 0.
        last_epoch (int): _LRScheduler에서 사용됨. 기본값: -1.
    """
    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=1e-7,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
