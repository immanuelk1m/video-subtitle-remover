import torch
import torch.nn as nn
import lpips
from model.vgg_arch import VGGFeatureExtractor

class PerceptualLoss(nn.Module):
    """일반적으로 사용되는 스타일 손실을 포함한 지각 손실.
 
    Args:
        layer_weights (dict): 각 vgg 특징 레이어에 대한 가중치.
            예시: {'conv5_4': 1.}는 conv5_4 특징 레이어(relu5_4 이전)가
            손실 계산 시 가중치 1.0으로 추출됨을 의미합니다.
        vgg_type (str): 특징 추출기로 사용되는 vgg 네트워크 유형.
            기본값: 'vgg19'.
        use_input_norm (bool): True이면 vgg에서 입력 이미지를 정규화합니다.
            기본값: True.
        range_norm (bool): True이면 [-1, 1] 범위의 이미지를 [0, 1]로 정규화합니다.
            기본값: False.
        perceptual_weight (float): `perceptual_weight > 0`이면 지각 손실이
            계산되고 손실에 가중치가 곱해집니다. 기본값: 1.0.
        style_weight (float): `style_weight > 0`이면 스타일 손실이
            계산되고 손실에 가중치가 곱해집니다.
            기본값: 0.
        criterion (str): 지각 손실에 사용되는 기준. 기본값: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """순방향 함수.
 
        Args:
            x (Tensor): (n, c, h, w) 모양의 입력 텐서.
            gt (Tensor): (n, c, h, w) 모양의 정답 텐서.
 
        Returns:
            Tensor: 순방향 결과.
        """
        # vgg 특징 추출
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # 지각 손실 계산
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # 스타일 손실 계산
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """그램 행렬 계산.
 
        Args:
            x (torch.Tensor): (n, c, h, w) 모양의 텐서.
 
        Returns:
            torch.Tensor: 그램 행렬.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class LPIPSLoss(nn.Module):
    def __init__(self, 
            loss_weight=1.0, 
            use_input_norm=True,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # 평균은 [0, 1] 범위의 이미지에 대한 값입니다.
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # 표준 편차는 [0, 1] 범위의 이미지에 대한 값입니다.
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return self.loss_weight * lpips_loss.mean(), None


class AdversarialLoss(nn.Module):
    r"""
    적대적 손실
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self,
                 type='nsgan',
                 target_real_label=1.0,
                 target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge (손실 유형)
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label
                      if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
