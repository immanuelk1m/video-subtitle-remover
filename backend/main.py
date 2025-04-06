import torch
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.tools.common_tools import is_video_or_image, is_image_file
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import multiprocessing
from shapely.geometry import Polygon
import time
from tqdm import tqdm
from tools.infer import utility
from tools.infer.predict_det import TextDetector


class SubtitleDetect:
    """
    텍스트 상자 감지 클래스, 비디오 프레임에 텍스트 상자가 있는지 감지하는 데 사용됩니다.
    """

    def __init__(self, video_path, sub_area=None):
        # 파라미터 객체 가져오기
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_PATH
        self.text_detector = TextDetector(args)
        self.video_path = video_path
        self.sub_area = sub_area

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        반환된 감지 상자에서 좌표를 가져옵니다.
        :param dt_box: 감지 상자 반환 결과
        :return list: 좌표점 목록
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 비디오 프레임 읽기 실패 (비디오 마지막 프레임까지 읽음)
            if not ret:
                break
            # 비디오 프레임 읽기 성공
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                if len(temp_list) > 0:
                    subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            # GUI 관련 진행률 업데이트 로직 제거
            # if sub_remover:
            #     sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        # if config.UNITE_COORDINATES:
        #     subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(subtitle_frame_no_box_dict)
        #     if sub_remover is not None:
        #         try:
        #             # 프레임 수가 1보다 크면 이미지나 단일 프레임이 아님을 의미
        #             if sub_remover.frame_count > 1:
        #                 subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
        #                                                                           sub_remover.fps)
        #         except Exception:
        #             pass
        #     subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 이산 값 목록이 정렬되었는지 확인
        points.sort()
        # 결과 구간을 저장하는 목록
        result_intervals = []
        # 구간 순회
        for start, end in intervals:
            # 현재 구간 내의 점
            current_points = [p for p in points if start <= p <= end]

            # 현재 구간 내의 이산 점 순회
            for p in current_points:
                # 현재 이산 점이 구간의 시작점이 아니면, 구간 시작부터 이산 점 이전 숫자까지의 구간 추가
                if start < p:
                    result_intervals.append((start, p - 1))
                # 구간 시작을 현재 이산 점으로 업데이트
                start = p
            # 마지막 이산 점 또는 구간 시작부터 구간 끝까지의 구간 추가
            result_intervals.append((start, end))
        # 결과 출력
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        장면 전환이 발생한 프레임 번호를 가져옵니다.
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """두 영역이 유사한지 판단합니다."""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """연속적으로 유사한 영역을 통합하고 목록 구조를 유지합니다."""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 키를 정렬하여 연속성을 보장합니다.
            unified_regions = {}

            # 초기화
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 매칭된 표준 구간을 저장할 새 목록 추가
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 현재 구간이 이전 키의 해당 구간과 유사하면 통합합니다.
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # unify_value_map을 최신 구간 값으로 업데이트
                unify_value_map[key] = new_unify_values
                last_key = key

            # 최종 통합된 결과를 unified_regions에 전달
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        자막이 나타나는 시작 프레임 번호와 끝 프레임 번호를 가져옵니다.
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 초기 구간 시작 값

        for i in range(1, len(numbers)):
            # 현재 숫자와 이전 숫자 간격이 1을 초과하면,
            # 이전 구간이 종료되고 현재 구간의 시작과 끝을 기록합니다.
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 이 숫자는 현재 연속 구간의 끝점입니다.
                ranges.append((start, end))
                start = numbers[i]  # 다음 연속 구간 시작
        # 마지막 구간 추가
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 초기 구간 시작 값
        for i in range(1, len(numbers)):
            # 현재 프레임 번호와 이전 프레임 번호 간격이 1을 초과하면,
            # 이전 구간이 종료되고 현재 구간의 시작과 끝을 기록합니다.
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 이 숫자는 현재 연속 구간의 끝점입니다.
                ranges.append((start, end))
                start = numbers[i]  # 다음 연속 구간 시작
            # 현재 프레임 번호와 이전 프레임 번호 간격이 1이고, 현재 프레임 번호에 해당하는 좌표점과 이전 프레임 번호에 해당하는 좌표점이 일치하지 않으면
            # 현재 구간의 시작과 끝을 기록합니다.
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 이 숫자는 현재 연속 구간의 끝점입니다.
                    ranges.append((start, end))
                    start = numbers[i]  # 다음 연속 구간 시작
        # 마지막 구간 추가
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # 출력 구간 목록 초기화
        expanded_intervals = []

        # 각 원본 구간 확장
        for interval in intervals:
            start, end = interval

            # 최소 'expand_size' 단위까지 확장하되, 'max_length' 단위를 초과하지 않음
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # 원본 구간 포함을 보장하면서 앞뒤 확장량을 최대한 균등하게 분배
            expand_start = max(start - expansion_amount // 2, 1)  # 시작점이 1보다 작지 않도록 보장
            expand_end = end + expansion_amount // 2

            # 확장된 구간이 최대 길이를 초과하면 조정
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # 단일 점 처리는 최소 'expand_size' 길이를 추가로 보장해야 함
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # 이전 구간과 겹치는지 확인하고 해당 병합 수행
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # 확장된 구간을 결과 목록에 추가
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        전달된 자막 시작 구간을 병합하고, 구간 크기가 최소 STTN_REFERENCE_LENGTH인지 확인합니다.
        """
        expanded = []
        # 먼저 단일 점 구간을 개별적으로 처리하여 확장합니다.
        for start, end in intervals:
            if start == end:  # 단일 점 구간
                # 목표 길이에 가깝게 확장하되, 앞뒤가 겹치지 않도록 보장합니다.
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # 다음 구간의 시작점 찾기
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # 새로운 확장 시작점과 끝점 결정
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # 새로운 확장 끝점이 시작점 앞에 있으면 확장할 공간이 충분하지 않음을 의미합니다.
                if new_end < new_start:
                    new_start, new_end = start, start  # 원래대로 유지
                expanded.append((new_start, new_end))
            else:
                # 비단일 점 구간은 직접 유지하고, 나중에 가능한 겹침 처리
                expanded.append((start, end))
        # 확장으로 인해 겹치는 구간을 병합하기 위해 정렬
        expanded.sort(key=lambda x: x[0])
        # 겹치는 구간을 병합하되, 실제로 겹치고 목표 길이보다 작은 경우에만 병합
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # 겹치는지 확인
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 병합 필요
                merged[-1] = (last_start, max(last_end, end))  # 구간 병합
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 인접 구간도 병합해야 하는 경우
                merged[-1] = (last_start, end)
            else:
                # 겹치지 않고 모두 목표 길이보다 크면 직접 유지
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # 면적이 가장 큰 텍스트 상자 찾기
            current_no = start_no
            # 현재 구간 사각형 상자의 최대 면적 찾기
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # 각 텍스트 상자 좌표 가져오기
                    xmin, xmax, ymin, ymax = coord
                    # 현재 텍스트 상자 좌표 면적 계산
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # 구간 최대 상자 목록이 비어 있으면 현재 면적이 구간 최대 면적
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # 목록이 비어 있지 않으면 현재 텍스트 상자가 구간 최대 텍스트 상자와 동일한 영역에 있는지 판단
                    else:
                        has_same_position = False
                        # 각 구간 최대 텍스트 상자를 순회하며 현재 텍스트 상자 위치가 구간 최대 텍스트 상자 목록의 특정 텍스트 상자와 동일한 행에 있고 교차하는지 판단
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # 높이 차이가 다르면
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # 동일한 행에 있으면 현재 면적이 최대인지 계산
                                    # 면적 크기를 판단하여 현재 면적이 더 크면 현재 행의 최대 영역 좌표점 업데이트
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # 모든 구간 최대 텍스트 상자 목록을 순회하고 새로운 행임을 발견하면 직접 추가
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        여러 비디오 프레임의 텍스트 영역 좌표를 통합합니다.
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        누락 감지를 방지하기 위해 추가 텍스트 상자를 추가합니다.
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        잘못된 자막 영역을 필터링합니다.
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


class SubtitleRemover:
    def __init__(self, vd_path, output_path, sub_area=None):
        importlib.reload(config)
        # 스레드 락
        self.lock = threading.RLock()
        # 사용자가 지정한 자막 영역 위치
        self.sub_area = sub_area
        # GUI 관련 속성 제거
        # self.gui_mode = gui_mode
        # 이미지인지 판단
        self.is_picture = False
        if is_image_file(str(vd_path)):
            self.sub_area = None
            self.is_picture = True
        # 비디오 경로
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 비디오 경로를 통해 비디오 이름 가져오기
        self.vd_name = Path(self.video_path).stem
        # 비디오 총 프레임 수
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 비디오 프레임 속도
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 비디오 크기
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 자막 감지 객체 생성
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 비디오 임시 객체 생성, Windows에서 delete=True는 permission denied 오류 발생 가능
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 비디오 쓰기 객체 생성
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        # 출력 경로는 생성자 인자로 받음 (한국어 주석)
        self.video_out_name = output_path
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            # 이미지 출력 경로도 인자로 받은 output_path 사용 (한국어 주석)
            # self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
            # 이미지인 경우 출력 디렉토리가 존재하는지 확인
            output_dir = os.path.dirname(self.video_out_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        # GUI 관련 속성 제거
        # self.progress_total = 0
        # self.progress_remover = 0
        self.isFinished = False
        # self.preview_frame = None
        # 원본 오디오를 자막 제거된 비디오에 포함할지 여부
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        반환된 감지 상자에서 좌표를 가져옵니다.
        :param dt_box: 감지 상자 반환 결과
        :return list: 좌표점 목록
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        주어진 프레임 번호가 시작인지 판단하고, 시작이면 끝 프레임 번호를 반환하고, 아니면 -1을 반환합니다.
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        주어진 프레임 번호가 시작인지 판단하고, 시작이면 끝 프레임 번호를 반환하고, 아니면 -1을 반환합니다.
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        # tqdm이 진행률을 표시하므로 내부 진행률 속성 업데이트 제거 (한국어 주석)
        tbar.update(increment)
        # current_percentage = (tbar.n / tbar.total) * 100
        # self.progress_remover = int(current_percentage) // 2
        # self.progress_total = 50 + self.progress_remover

    def propainter_mode(self, tbar):
        print('use propainter mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            # 현재 프레임에 워터마크/텍스트가 없으면 직접 쓰기
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                print(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                continue
            # 워터마크가 있으면 해당 프레임이 시작 프레임인지 판단
            else:
                # 시작 프레임이면 끝 프레임까지 일괄 추론
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # print(f'No 1 Current index: {index}')
                    start_frame_no = index
                    print(f'find start: {start_frame_no}')
                    # 끝 프레임 찾기
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 현재 프레임 번호가 자막 시작 위치인지 판단
                    # 가져온 끝 프레임 번호가 -1이 아니면 설명
                    if end_frame_no != -1:
                        print(f'find end: {end_frame_no}')
                        # ************ 해당 구간 모든 프레임 읽기 시작 ************
                        temp_frames = list()
                        # 첫 프레임을 처리 목록에 추가
                        temp_frames.append(frame)
                        inner_index = 0
                        # 끝 프레임까지 계속 읽기
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 해당 구간 모든 프레임 읽기 끝 ************
                        if len(temp_frames) < 1:
                            # 처리할 내용 없음, 직접 건너뛰기
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 읽은 비디오 프레임을 일괄 처리
                            # 1. 현재 배치에서 사용할 마스크 가져오기
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, config.PROPAINTER_MAX_LOAD_NUM):
                                # 2. 일괄 추론 호출
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = self.video_inpaint.inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        # GUI 미리보기 로직 제거
                                        # if self.gui_mode:
                                        #     self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                                self.update_progress(tbar, increment=len(batch))

    def sttn_mode_with_no_detection(self, tbar):
        """
        sttn을 사용하여 선택한 영역을 다시 그리고, 자막 감지는 수행하지 않습니다.
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        # input_sub_remover 인자 제거 (GUI 진행률 업데이트 불필요) (한국어 주석)
        sttn_video_inpaint(input_mask=mask, tbar=tbar)

    def sttn_mode(self, tbar):
        # 자막 프레임 찾기 건너뛰기 여부
        if config.STTN_SKIP_DETECTION:
            # 건너뛰면 sttn 모드 사용
            self.sttn_mode_with_no_detection(tbar)
        else:
            print('use sttn mode')
            sttn_inpaint = STTNInpaint()
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            print(continuous_frame_no_list)
            continuous_frame_no_list = self.sub_detector.filter_and_merge_intervals(continuous_frame_no_list)
            print(continuous_frame_no_list)
            start_end_map = dict()
            for interval in continuous_frame_no_list:
                start, end = interval
                start_end_map[start] = end
            current_frame_index = 0
            print('[Processing] start removing subtitles...')
            while True:
                ret, frame = self.video_cap.read()
                # 읽기가 완료되면 종료
                if not ret:
                    break
                current_frame_index += 1
                # 현재 프레임 번호가 자막 구간 시작인지 판단, 아니면 직접 쓰기
                if current_frame_index not in start_end_map.keys():
                    self.video_writer.write(frame)
                    print(f'write frame: {current_frame_index}')
                    self.update_progress(tbar, increment=1)
                    # GUI 미리보기 로직 제거
                    # if self.gui_mode:
                    #     self.preview_frame = cv2.hconcat([frame, frame])
                # 구간 시작이면 끝 찾기
                else:
                    start_frame_index = current_frame_index
                    end_frame_index = start_end_map[current_frame_index]
                    print(f'processing frame {start_frame_index} to {end_frame_index}')
                    # 자막 제거가 필요한 비디오 프레임을 저장하는 데 사용
                    frames_need_inpaint = list()
                    frames_need_inpaint.append(frame)
                    inner_index = 0
                    # 끝까지 계속 읽기
                    for j in range(end_frame_index - start_frame_index):
                        ret, frame = self.video_cap.read()
                        if not ret:
                            break
                        current_frame_index += 1
                        frames_need_inpaint.append(frame)
                    mask_area_coordinates = []
                    # 1. 현재 배치의 마스크 좌표 전체 집합 가져오기
                    for mask_index in range(start_frame_index, end_frame_index):
                        if mask_index in sub_list.keys():
                            for area in sub_list[mask_index]:
                                xmin, xmax, ymin, ymax = area
                                # 비자막 영역인지 판단 (너비가 높이보다 크면 잘못된 감지로 간주)
                                if (ymax - ymin) - (xmax - xmin) > config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE:
                                    continue
                                if area not in mask_area_coordinates:
                                    mask_area_coordinates.append(area)
                    # 1. 현재 배치에서 사용할 마스크 가져오기
                    mask = create_mask(self.mask_size, mask_area_coordinates)
                    print(f'inpaint with mask: {mask_area_coordinates}')
                    for batch in batch_generator(frames_need_inpaint, config.STTN_MAX_LOAD_NUM):
                        # 2. 일괄 추론 호출
                        if len(batch) >= 1:
                            inpainted_frames = sttn_inpaint(batch, mask)
                            for i, inpainted_frame in enumerate(inpainted_frames):
                                self.video_writer.write(inpainted_frame)
                                print(f'write frame: {start_frame_index + inner_index} with mask')
                                inner_index += 1
                                # GUI 미리보기 로직 제거
                                # if self.gui_mode:
                                #     self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                        self.update_progress(tbar, increment=len(batch))

    def lama_mode(self, tbar):
        print('use lama mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        index = 0
        print('[Processing] start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list.keys():
                mask = create_mask(self.mask_size, sub_list[index])
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
            # GUI 미리보기 로직 제거
            # if self.gui_mode:
            #     self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            tbar.update(1)
            # GUI 관련 진행률 업데이트 로직 제거
            # self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            # self.progress_total = 50 + self.progress_remover

    def run(self):
        # 시작 시간 기록
        start_time = time.time()
        # 진행률 표시줄 재설정
        # GUI 관련 진행률 속성 제거
        # self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.lama_inpaint = LamaInpaint()
            original_frame = cv2.imread(self.video_path)
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list[1])
                inpainted_frame = self.lama_inpaint(original_frame, mask)
            else:
                inpainted_frame = original_frame
            # GUI 미리보기 로직 제거
            # if self.gui_mode:
            #     self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
            tbar.update(1)
            # GUI 관련 진행률 속성 제거
            # self.progress_total = 100
        else:
            # 정밀 모드에서 장면 분할 프레임 번호를 가져와 추가 분할
            if config.MODE == config.InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.MODE == config.InpaintMode.STTN:
                self.sttn_mode(tbar)
            else:
                self.lama_mode(tbar)
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 원본 오디오를 새로 생성된 비디오 파일에 병합
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        # GUI 관련 진행률 속성 제거
        # self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

    def merge_audio_to_video(self):
        # 오디오 임시 객체 생성, Windows에서 delete=True는 permission denied 오류 발생 가능
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell)
        except Exception:
            print('fail to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [config.FFMPEG_PATH,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "libx264" if config.USE_H264 else "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_name]
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell)
                except Exception:
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() in ['Windows']:
                        pass
                    else:
                        print(f'failed to delete temp file {temp.name}')
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()


def parse_sub_area(area_str):
    """'ymin,ymax,xmin,xmax' 형식의 자막 영역 문자열을 정수 튜플로 파싱합니다."""
    try:
        coords = [int(c.strip()) for c in area_str.split(',')]
        if len(coords) == 4:
            # ymin, ymax, xmin, xmax
            return coords[0], coords[1], coords[2], coords[3]
        else:
            raise ValueError("Subtitle area must be in 'ymin,ymax,xmin,xmax' format.")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid subtitle area format: {area_str}. Error: {e}")

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description='비디오나 이미지에서 자막을 제거합니다.')
    parser.add_argument('--input', type=str, required=True, help='입력 비디오 또는 이미지 파일 경로.')
    parser.add_argument('--output', type=str, required=True, help='출력 파일을 저장할 경로.')
    parser.add_argument('--sub_area', type=parse_sub_area, default=None,
                        help="'ymin,ymax,xmin,xmax' 형식의 자막 영역 좌표 (예: '720,900,100,1820'). "
                             "지정하지 않으면 프로그램이 자동으로 감지하거나 전체 프레임을 처리합니다.")
    parser.add_argument('--mode', type=str, default='propainter', choices=['lama', 'sttn', 'propainter'],
                        help='사용할 인페인팅 모드 (lama, sttn, propainter). 기본값은 propainter입니다.')

    cli_args = parser.parse_args()

    # CLI 인자를 기반으로 config.MODE 설정
    if cli_args.mode == 'sttn':
        config.MODE = config.InpaintMode.STTN
    elif cli_args.mode == 'propainter':
        config.MODE = config.InpaintMode.PROPAINTER
    else: # 기본값은 lama
        config.MODE = config.InpaintMode.LAMA

    # 입력 경로 유효성 검사
    if not os.path.exists(cli_args.input):
        print(f"Error: Input file not found at {cli_args.input}")
        sys.exit(1)
    if not is_video_or_image(cli_args.input):
        print(f"Error: Invalid input file type at {cli_args.input}. Only video and image files are supported.")
        sys.exit(1)

    # SubtitleRemover 인스턴스 생성 및 실행
    try:
        sd = SubtitleRemover(vd_path=cli_args.input, output_path=cli_args.output, sub_area=cli_args.sub_area)
        sd.run()
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
