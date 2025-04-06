import copy
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

# 이미지 전처리 방식 정의
_to_tensors = transforms.Compose([
    Stack(),  # 이미지를 시퀀스로 스택
    ToTorchFormatTensor()  # 스택된 이미지를 PyTorch 텐서로 변환
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. InpaintGenerator 모델 인스턴스를 생성하고 선택한 장치에 로드
        self.model = InpaintGenerator().to(self.device)
        # 2. 사전 훈련된 모델의 가중치를 로드하고 모델 상태 사전 전재
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location=self.device)['netG'])
        # 3. 모델을 평가 모드로 설정
        self.model.eval()
        # 모델 입력에 사용되는 너비와 높이
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 연결된 프레임 수 설정
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 원본 비디오 프레임
        :param mask: 자막 영역 마스크
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 자막 제거할 세로 높이 부분 결정
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # 프레임 저장 변수 초기화
        # 고해상도 프레임 저장 목록
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # 스케일링된 프레임을 저장하는 사전
        comps = {}  # 완성된 프레임을 저장하는 사전
        # 최종 비디오 프레임 저장
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 각 제거 부분에 대한 목록 초기화

        # 프레임 읽기 및 스케일링
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 각 제거 부분에 대해 자르기 및 스케일링 수행
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 자르기
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 스케일링
                frames_scaled[k].append(image_resize)  # 스케일링된 프레임을 해당 목록에 추가

        # 각 제거 부분 처리
        for k in range(len(inpaint_area)):
            # inpaint 함수를 호출하여 처리
            comps[k] = self.inpaint(frames_scaled[k])

        # 제거 부분이 있는 경우
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 원본 프레임 가져오기
                # 모드의 각 단락에 대해
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h))  # 완성된 프레임을 원본 크기로 다시 스케일링
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # 색 공간 변환
                    # 마스크 영역 가져오기 및 이미지 합성 수행
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # 마스크 영역 가져오기
                    # 마스크 영역 내 이미지 융합 구현
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # 최종 프레임을 목록에 추가
                inpainted_frames.append(frame)
                print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 이진 마스크로 변환
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        전체 비디오의 참조 프레임 샘플링
        """
        # 참조 프레임 인덱스 목록 초기화
        ref_index = []
        # 비디오 길이 범위 내에서 ref_length에 따라 단계적으로 반복
        for i in range(0, length, self.ref_length):
            # 현재 프레임이 인접 프레임에 없으면
            if i not in neighbor_ids:
                # 참조 프레임 목록에 추가
                ref_index.append(i)
        # 참조 프레임 인덱스 목록 반환
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        STTN을 사용하여 구멍 채우기 (구멍은 마스크된 영역)
        """
        frame_length = len(frames)
        # 프레임을 전처리하여 텐서로 변환하고 정규화
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 특징 텐서를 지정된 장치(CPU 또는 GPU)로 전송
        feats = feats.to(self.device)
        # 비디오 길이와 동일한 목록을 초기화하여 처리 완료된 프레임 저장
        comp_frames = [None] * frame_length
        # 추론 단계에서 메모리 절약 및 가속화를 위해 그래디언트 계산 비활성화
        with torch.no_grad():
            # 처리된 프레임을 인코더를 통해 특징 표현 생성
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 특징 차원 정보 가져오기
            _, c, feat_h, feat_w = feats.size()
            # 모델의 예상 입력과 일치하도록 특징 모양 조정
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # 다시 그릴 영역 가져오기
        # 설정된 인접 프레임 간격 내에서 비디오 순환 처리
        for f in range(0, frame_length, self.neighbor_stride):
            # 인접 프레임 ID 계산
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 참조 프레임 인덱스 가져오기
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # 마찬가지로 그래디언트 계산 비활성화
            with torch.no_grad():
                # 모델을 통해 특징을 추론하고 디코더에 전달하여 완성된 프레임 생성
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # 예측된 특징을 디코더를 통해 이미지로 생성하고 활성화 함수 tanh 적용 후 텐서 분리
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # 결과 텐서를 0에서 255 범위(이미지 픽셀 값)로 다시 스케일링
                pred_img = (pred_img + 1) / 2
                # 텐서를 CPU로 다시 이동하고 NumPy 배열로 변환
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 인접 프레임 순회
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 예측된 이미지를 부호 없는 8비트 정수 형식으로 변환
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # 해당 위치가 비어 있으면 새로 계산된 이미지로 할당
                        comp_frames[idx] = img
                    else:
                        # 이 위치에 이전에 이미지가 있으면 새 이미지와 이전 이미지를 혼합하여 품질 향상
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 처리 완료된 프레임 시퀀스 반환
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        자막 제거 영역을 가져오고, 마스크를 기반으로 채워야 할 영역과 높이를 결정합니다.
        """
        # 그리기 영역 저장 목록
        inpaint_area = []
        # 비디오 하단의 자막 위치에서 시작, 자막은 일반적으로 하단에 있다고 가정
        to_H = from_H = H
        # 하단에서 위로 마스크 순회
        while from_H != 0:
            if to_H - h < 0:
                # 다음 단락이 상단을 초과하면 상단에서 시작
                from_H = 0
                to_H = h
            else:
                # 단락의 상단 경계 결정
                from_H = to_H - h
            # 현재 단락에 마스크 픽셀이 포함되어 있는지 확인
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # 첫 번째 단락이 아니면 아래로 이동하여 마스크 영역 누락 방지
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # 하단을 넘지 않도록 보장
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # 해당 단락을 목록에 추가
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            # 다음 단락으로 이동
            to_H -= h
        return inpaint_area  # 그리기 영역 목록 반환

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # 결과 저장 목록
        inpaint_area = []
        # 표준 구간 계산 및 저장
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # 마지막 구간이 최대값에 도달했는지 확인
        if inpaint_area[-1][1] != ymax:
            # 그렇지 않으면 마지막 구간의 끝에서 시작하여 확장된 값으로 끝나는 새 구간 생성
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # 그리기 영역 목록 반환


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # opencv를 사용하여 비디오 읽기
        reader = cv2.VideoCapture(self.video_path)
        # 비디오의 너비, 높이, 프레임 속도 및 프레임 수 정보를 가져와 frame_info 사전에 저장
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 비디오 원본 너비
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 비디오 원본 높이
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 비디오 프레임 속도
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 비디오 총 프레임 수
        }
        # 비디오 읽기 객체, 프레임 정보 및 비디오 쓰기 객체 반환
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None):
        # STTNInpaint 비디오 복원 인스턴스 초기화
        self.sttn_inpaint = STTNInpaint()
        # 비디오 및 마스크 경로
        self.video_path = video_path
        self.mask_path = mask_path
        # 출력 비디오 파일 경로 설정
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # 한 번의 처리에서 로드할 수 있는 최대 프레임 수 구성
        if clip_gap is None:
            self.clip_gap = config.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        # 비디오 프레임 정보 읽기
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            # 복원된 비디오를 출력하기 위한 비디오 쓰기 객체 생성
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
        # 비디오 반복 복원에 필요한 횟수 계산
        rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
        # 복원 영역 크기를 결정하는 데 사용되는 분할 높이 계산
        split_h = int(frame_info['W_ori'] * 3 / 16)
        if input_mask is None:
            # 마스크 읽기
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        # 복원 영역 위치 가져오기
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(frame_info['H_ori'], split_h, mask)
        # 각 반복 횟수 순회
        for i in range(rec_time):
            start_f = i * self.clip_gap  # 시작 프레임 위치
            end_f = min((i + 1) * self.clip_gap, frame_info['len'])  # 끝 프레임 위치
            print('Processing:', start_f + 1, '-', end_f, ' / Total:', frame_info['len'])
            frames_hr = []  # 고해상도 프레임 목록
            frames = {}  # 프레임 사전, 잘라낸 이미지 저장용
            comps = {}  # 조합 사전, 복원된 이미지 저장용
            # 프레임 사전 초기화
            for k in range(len(inpaint_area)):
                frames[k] = []
            # 고해상도 프레임 읽기 및 복원
            for j in range(start_f, end_f):
                success, image = reader.read()
                frames_hr.append(image)
                for k in range(len(inpaint_area)):
                    # 자르기, 스케일링 및 프레임 사전에 추가
                    image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                    frames[k].append(image_resize)
            # 각 복원 영역에 대해 복원 실행
            for k in range(len(inpaint_area)):
                comps[k] = self.sttn_inpaint.inpaint(frames[k])
            # 복원할 영역이 있는 경우
            if inpaint_area is not []:
                for j in range(end_f - start_f):
                    if input_sub_remover is not None and input_sub_remover.gui_mode:
                        original_frame = copy.deepcopy(frames_hr[j])
                    else:
                        original_frame = None
                    frame = frames_hr[j]
                    for k in range(len(inpaint_area)):
                        # 복원된 이미지를 원본 해상도로 다시 확장하고 원본 프레임에 융합
                        comp = cv2.resize(comps[k][j], (frame_info['W_ori'], split_h))
                        comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                        frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    writer.write(frame)
                    if input_sub_remover is not None:
                        if tbar is not None:
                            input_sub_remover.update_progress(tbar, increment=1)
                        if original_frame is not None and input_sub_remover.gui_mode:
                            input_sub_remover.preview_frame = cv2.hconcat([original_frame, frame])
        # 비디오 쓰기 객체 해제
        writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # 시작 시간 기록
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
