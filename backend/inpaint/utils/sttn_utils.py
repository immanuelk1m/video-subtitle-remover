import matplotlib.patches as patches
from matplotlib.path import Path
import io
import cv2
import random
import zipfile
import numpy as np
from PIL import Image, ImageOps

import torch

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')


class ZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = ZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = ZipReader.build_file_dict(path)
        znames = zfile.namelist()
        znames.sort()
        data = zfile.read(znames[idx])
        im = Image.open(io.BytesIO(data))
        return im

# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """주어진 PIL.Image를 0.5의 확률로 무작위로 수평 뒤집기합니다.
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    # 뒤집을 때 플로우 픽셀 값 반전
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        for i in range(len(img_group)):
            if img_group[i].ndim==3:
                img_group[i] = Image.fromarray(cv2.cvtColor(img_group[i], cv2.COLOR_BGR2RGB))
            elif img_group[i].ndim==2:
                img_group[i] = Image.fromarray(img_group[i])

        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ [0, 255] 범위의 PIL.Image (RGB) 또는 numpy.ndarray (H x W x C)를
    [0.0, 1.0] 범위의 (C x H x W) 모양의 torch.FloatTensor로 변환합니다. """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy 이미지: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # PIL 이미지 처리
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # HWC에서 CHW 형식으로 변경
            # 이런, 이 전치는 로딩 시간/CPU의 80%를 차지합니다.
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################

def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # 무작위 모양 가져오기
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # 무작위 위치 가져오기
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # 고정 마스크 반환
    if random.uniform(0, 1) > 0.5:
        return masks*video_length
    # 움직이는 마스크 반환
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      초기점과 3차 베지어 곡선당 3개의 점이 있습니다.
      따라서 곡선은 날카로운 모서리가 될 n개의 점만 통과합니다.
      다른 2개는 베지어 곡선의 모양을 수정합니다.
      edge_num, 가능한 날카로운 모서리 수
      points_num, 경로의 점 수
      ratio, (0, 1) 단위 원에서의 섭동 크기,
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Path.CLOSEPOLY 대신 이것을 사용하면 불필요한 직선을 피할 수 있습니다.
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # 경로를 이미지로 그리기
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # 모양만 남기기 위해 축 제거
    fig.canvas.draw()
    # plt 이미지를 numpy 이미지로 변환
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # 후처리
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity


if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # 반환된 마스크는 정지(50%) 또는 이동(50%)입니다.
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432)
        print(np.array(masks[0]).shape)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)

