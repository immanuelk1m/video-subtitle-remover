"""
https://arxiv.org/abs/1802.05957 논문의 스펙트럼 정규화
"""
import torch
from torch.nn.functional import normalize


class SpectralNorm(object):
    # 각 forward 호출 전후의 불변성:
    #   u = normalize(W @ v)
    # 참고: 초기화 시에는 이 불변성이 강제되지 않습니다.

    _version = 1
    # 버전 1에서:
    #   `W`를 버퍼가 아니도록 만들었습니다.
    #   `v`를 버퍼로 추가했습니다.
    #   평가 모드에서 저장된 `W` 대신 `W = u @ W_orig @ v`를 사용하도록 만들었습니다.

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # 차원을 앞으로 순열
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        # 참고: `do_power_iteration`이 설정되면 `u`와 `v` 벡터는
        #     거듭제곱 반복에서 **제자리에서** 업데이트됩니다. 이는 매우 중요합니다.
        #     왜냐하면 `DataParallel` forward에서 벡터(버퍼임)는
        #     병렬화된 모듈에서 각 모듈 복제본으로 브로드캐스트되기 때문입니다.
        #     이 복제본은 즉석에서 생성되는 새로운 모듈 객체입니다. 그리고 각 복제본은
        #     자체 스펙트럼 정규화 거듭제곱 반복을 실행합니다. 따라서 단순히
        #     업데이트된 벡터를 이 함수가 실행되는 모듈에 할당하면
        #     업데이트가 영원히 손실됩니다. 그리고 다음에 병렬화된
        #     모듈이 복제될 때, 동일한 무작위로 초기화된 벡터가
        #     브로드캐스트되어 사용됩니다!
        #
        #     따라서 변경 사항을 다시 전파하려면 두 가지
        #     중요한 동작(테스트를 통해 강제됨)에 의존합니다.
        #       1. `DataParallel`은 브로드캐스트 텐서가
        #          이미 올바른 장치에 있는 경우 스토리지를 복제하지 않습니다. 그리고
        #          병렬화된 모듈이 이미 `device[0]`에 있는지 확인합니다.
        #       2. `out=` kwarg의 out 텐서가 올바른 모양을 가지면
        #          값을 채우기만 합니다.
        #     따라서 모든 장치에서 동일한 거듭제곱 반복이 수행되므로
        #     텐서를 제자리에서 업데이트하기만 하면
        #     `device[0]`의 모듈 복제본이 병렬화된 모듈의 _u 벡터를
        #     (공유 스토리지를 통해) 업데이트하도록 보장합니다.
        #
        #    그러나 `u`와 `v`를 제자리에서 업데이트한 후에는 가중치를 정규화하기 전에
        #    **복제**해야 합니다. 이는 두 번의 forward 패스를 통한 역전파를 지원하기 위함입니다.
        #    예: GAN 훈련의 일반적인 패턴: loss = D(real) - D(fake). 그렇지 않으면 엔진은
        #    첫 번째 forward에 대한 역전파에 필요한 변수(즉, `u` 및 `v` 벡터)가
        #    두 번째 forward에서 변경되었다고 불평할 것입니다.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # 가중치의 스펙트럼 노름은 `u^T W v`와 같습니다. 여기서 `u`와 `v`는
                    # 첫 번째 왼쪽 및 오른쪽 특이 벡터입니다.
                    # 이 거듭제곱 반복은 `u`와 `v`의 근사치를 생성합니다.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # 복제해야 하는 이유에 대해서는 위를 참조하십시오.
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # `u = normalize(W @ v)` (이 클래스 상단의 불변성) 및 `u @ W @ v = sigma`를 만족하는
        # 벡터 `v`를 반환하려고 시도합니다.
        # W^T W가 역행렬이 아닌 경우 pinverse를 사용합니다.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # `u`와 `v`를 무작위로 초기화
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # 여전히 가중치를 fn.name으로 다시 할당해야 합니다. 왜냐하면 모든 종류의
        # 것들이 그것이 존재한다고 가정할 수 있기 때문입니다. 예: 가중치 초기화 시.
        # 그러나 nn.Parameter일 수 있고 파라미터로 추가되기 때문에 직접 할당할 수 없습니다.
        # 대신 weight.data를 일반 속성으로 등록합니다.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


# Py2 pickle은 내부 클래스나 인스턴스 메서드를 좋아하지 않으므로 최상위 클래스입니다.
class SpectralNormLoadStateDictPreHook(object):
    # spectral_norm 변경 사항에 대한 SpectralNorm._version의 docstring 참조.
    def __init__(self, fn):
        self.fn = fn

    # 버전 None의 state_dict의 경우 (최소 한 번의 훈련 forward를 거쳤다고 가정),
    # 다음이 성립합니다.
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, 여기서 sigma = u @ W_orig @ v
    #
    # `v`를 계산하기 위해 `W_orig @ x = u`를 풀고,
    #    v = x / (u @ W_orig @ x) * (W / W_orig)로 둡니다.
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                # weight = state_dict.pop(prefix + fn.name)
                # sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']
                # v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                # state_dict[prefix + fn.name + '_v'] = v


# Py2 pickle은 내부 클래스나 인스턴스 메서드를 좋아하지 않으므로 최상위 클래스입니다.
class SpectralNormStateDictHook(object):
    # spectral_norm 변경 사항에 대한 SpectralNorm._version의 docstring 참조.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""주어진 모듈의 파라미터에 스펙트럼 정규화를 적용합니다.
 
    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
 
    스펙트럼 정규화는 생성적 적대 신경망(GAN)에서 판별자(비평가)의 훈련을 안정화합니다.
    이는 거듭제곱 반복 방법을 사용하여 계산된 가중치 행렬의 스펙트럼 노름 :math:`\sigma`로
    가중치 텐서를 재조정함으로써 이루어집니다. 가중치 텐서의 차원이 2보다 크면
    스펙트럼 노름을 얻기 위해 거듭제곱 반복 방법에서 2D로 재구성됩니다.
    이는 스펙트럼 노름을 계산하고 모든 :meth:`~Module.forward` 호출 전에 가중치를 재조정하는
    훅을 통해 구현됩니다.

    `Spectral Normalization for Generative Adversarial Networks`_ 논문을 참조하십시오.

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): 포함하는 모듈
        name (str, optional): 가중치 파라미터 이름
        n_power_iterations (int, optional): 스펙트럼 노름을 계산하기 위한
            거듭제곱 반복 횟수
        eps (float, optional): 노름 계산 시 수치적 안정성을 위한 엡실론
        dim (int, optional): 출력 수에 해당하는 차원,
            기본값은 ``0``이며, ConvTranspose{1,2,3}d 인스턴스인 모듈의 경우 ``1``입니다.

    Returns:
        스펙트럼 노름 훅이 있는 원본 모듈

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""모듈에서 스펙트럼 정규화 재파라미터화를 제거합니다.

    Args:
        module (Module): 포함하는 모듈
        name (str, optional): 가중치 파라미터 이름

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))


def use_spectral_norm(module, use_sn=False):
    if use_sn:
        return spectral_norm(module)
    return module