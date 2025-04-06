import numpy as np
from skimage import measure
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import to_tensors


def calculate_epe(flow1, flow2):
    """종점 오차(End point errors)를 계산합니다."""

    epe = torch.sum((flow1 - flow2)**2, dim=1).sqrt()
    epe = epe.view(-1)
    return epe.mean().item()


def calculate_psnr(img1, img2):
    """PSNR(최대 신호 대 잡음비)을 계산합니다.
    참조: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): [0, 255] 범위의 이미지.
        img2 (ndarray): [0, 255] 범위의 이미지.
    Returns:
        float: PSNR 결과.
    """

    assert img1.shape == img2.shape, \
        (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calc_psnr_and_ssim(img1, img2):
    """이미지에 대한 PSNR 및 SSIM을 계산합니다.
        img1: ndarray, 범위 [0, 255]
        img2: ndarray, 범위 [0, 255]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    psnr = calculate_psnr(img1, img2)
    ssim = measure.compare_ssim(img1,
                                img2,
                                data_range=255,
                                multichannel=True,
                                win_size=65)

    return psnr, ssim


###########################
# I3D 모델
###########################


def init_i3d_model(i3d_model_path):
    print(f"[Loading I3D model from {i3d_model_path} for FID score ..]")
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_path))
    i3d_model.to(torch.device('cuda:0'))
    return i3d_model


def calculate_i3d_activations(video1, video2, i3d_model, device):
    """VFID 메트릭을 계산합니다.
        video1: list[PIL.Image]
        video2: list[PIL.Image]
    """
    video1 = to_tensors()(video1).unsqueeze(0).to(device)
    video2 = to_tensors()(video2).unsqueeze(0).to(device)
    video1_activations = get_i3d_activations(
        video1, i3d_model).cpu().numpy().flatten()
    video2_activations = get_i3d_activations(
        video2, i3d_model).cpu().numpy().flatten()

    return video1_activations, video2_activations


def calculate_vfid(real_activations, fake_activations):
    """
    두 특징 분포가 주어졌을 때, 그 사이의 FID 점수를 계산합니다.
    Params:
        real_activations: list[ndarray] (실제 활성화 값 목록)
        fake_activations: list[ndarray] (가짜 활성화 값 목록)
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """프레셰 거리(Frechet Distance)의 Numpy 구현.
    두 다변량 가우시안 X_1 ~ N(mu_1, C_1)과 X_2 ~ N(mu_2, C_2) 사이의 프레셰 거리는 다음과 같습니다.
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Dougal J. Sutherland의 안정적인 버전.
    Params:
    -- mu1   : 생성된 샘플에 대한 인셉션 넷 레이어의 활성화 값을 포함하는 Numpy 배열
               ('get_predictions' 함수에서 반환되는 것과 유사).
    -- mu2   : 대표적인 데이터 세트에서 미리 계산된 활성화 값에 대한 샘플 평균.
    -- sigma1: 생성된 샘플에 대한 활성화 값의 공분산 행렬.
    -- sigma2: 대표적인 데이터 세트에서 미리 계산된 활성화 값의 공분산 행렬.
    Returns:
    --   : 프레셰 거리.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # 곱이 거의 특이 행렬일 수 있음
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 수치 오류로 인해 약간의 허수 성분이 발생할 수 있음
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA (코드 분석 비활성화 주석)
            np.trace(sigma2) - 2 * tr_covmean)


def get_i3d_activations(batched_video,
                        i3d_model,
                        target_endpoint='Logits',
                        flatten=True,
                        grad_enabled=False):
    """
    i3d 모델에서 특징을 가져와 1차원 특징으로 평탄화합니다.
    유효한 대상 엔드포인트는 InceptionI3d.VALID_ENDPOINTS에 정의되어 있습니다.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2),
                                          target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat


# 이 코드는 https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py 에서 가져왔습니다.
# 여기서는 flake8 오류 수정 및 일부 정리만 수행했습니다.


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # 'same' 패딩 계산
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self,
                 in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Unit3D 모듈을 초기화합니다."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # 여기서는 항상 패딩을 0으로 설정합니다. forward 함수에서
            # 입력 크기에 따라 동적으로 패딩합니다.
            bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels,
                                     eps=0.001,
                                     momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # 'same' 패딩 계산
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels,
                         output_channels=out_channels[0],
                         kernel_shape=[1, 1, 1],
                         padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[1],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1],
                          output_channels=out_channels[2],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[3],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3],
                          output_channels=out_channels[4],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1),
                                        padding=0)
        self.b3b = Unit3D(in_channels=in_channels,
                          output_channels=out_channels[5],
                          kernel_shape=[1, 1, 1],
                          padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D 아키텍처.
    이 모델은 다음 논문에서 소개되었습니다:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    다음 논문에서 소개된 Inception 아키텍처도 참조하십시오:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # 모델의 엔드포인트 순서. 구성 중 지정된 `final_endpoint`까지의 모든 엔드포인트는
    # 두 번째 반환 값으로 사전에 반환됩니다.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint='Logits',
                 name='inception_i3d',
                 in_channels=3,
                 dropout_keep_prob=0.5):
        """I3D 모델 인스턴스를 초기화합니다.
        Args:
          num_classes: 로짓 레이어의 출력 수 (기본값 400, Kinetics 데이터셋과 일치).
          spatial_squeeze: 반환하기 전에 로짓의 공간 차원을 축소할지 여부 (기본값 True).
          final_endpoint: 모델에는 많은 가능한 엔드포인트가 포함되어 있습니다.
              `final_endpoint`는 모델이 빌드될 마지막 엔드포인트를 지정합니다.
              `final_endpoint`에서의 출력 외에도 `final_endpoint`까지의 모든 엔드포인트에서의
              출력도 사전에 반환됩니다. `final_endpoint`는
              InceptionI3d.VALID_ENDPOINTS 중 하나여야 합니다 (기본값 'Logits').
          name: 문자열 (선택 사항). 이 모듈의 이름.
        Raises:
          ValueError: `final_endpoint`가 인식되지 않는 경우.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' %
                             self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels,
                                            output_channels=64,
                                            kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2),
                                            padding=(3, 3, 3),
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=64,
                                            kernel_shape=[1, 1, 1],
                                            padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64,
                                            output_channels=192,
                                            kernel_shape=[3, 3, 3],
                                            padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192,
                                                     [64, 96, 128, 16, 32, 32],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
            name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
                             output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128,
                             output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point]( # dataparallel과 함께 작동하도록 _modules 사용
                    x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits는 배치 X 시간 X 클래스이며, 이것이 우리가 작업하려는 형식입니다.
        return logits

    def extract_features(self, x, target_endpoint='Logits'):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
                if end_point == target_endpoint:
                    break
        if target_endpoint == 'Logits':
            return x.mean(4).mean(3).mean(2)
        else:
            return x
