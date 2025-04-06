import multiprocessing
import cv2
import numpy as np

from backend import config
from backend.inpaint.lama_inpaint import LamaInpaint


def batch_generator(data, max_batch_size):
    """
    데이터 크기에 따라 최대 길이가 max_batch_size를 초과하지 않는 균일한 배치 데이터를 생성합니다.
    """
    n_samples = len(data)
    # 모든 배치 크기가 최대한 비슷해지도록 MAX_BATCH_SIZE보다 작은 batch_size를 찾으려고 시도합니다.
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 마지막 배치가 batch_size보다 작을 수 있는 경우 처리
    # 마지막 배치가 다른 배치보다 적으면 batch_size를 줄여 각 배치의 수량을 균형 있게 조정하려고 시도합니다.
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 배치 크기 줄이기
        num_batches = n_samples // batch_size

    # 처음 num_batches개의 배치 생성
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 남은 데이터를 마지막 배치로 사용
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]


def inference_task(batch_data):
    inpainted_frame_dict = dict()
    for data in batch_data:
        index, original_frame, coords_list = data
        mask_size = original_frame.shape[:2]
        mask = create_mask(mask_size, coords_list)
        inpaint_frame = inpaint(original_frame, mask)
        inpainted_frame_dict[index] = inpaint_frame
    return inpainted_frame_dict


def parallel_inference(inputs, batch_size=None, pool_size=None):
    """
    병렬 추론을 수행하고 결과 순서를 유지합니다.
    """
    if pool_size is None:
        pool_size = multiprocessing.cpu_count()
    # 컨텍스트 관리자를 사용하여 프로세스 풀 자동 관리
    with multiprocessing.Pool(processes=pool_size) as pool:
        batched_inputs = list(batch_generator(inputs, batch_size))
        # map 함수를 사용하여 입력과 출력 순서가 일치하도록 보장
        batch_results = pool.map(inference_task, batched_inputs)
    # 배치 추론 결과 펼치기
    index_inpainted_frames = [item for sublist in batch_results for item in sublist]
    return index_inpainted_frames


def inpaint(img, mask):
    lama_inpaint_instance = LamaInpaint()
    img_inpainted = lama_inpaint_instance(img, mask)
    return img_inpainted


def inpaint_with_multiple_masks(censored_img, mask_list):
    inpainted_frame = censored_img
    if mask_list:
        for mask in mask_list:
            inpainted_frame = inpaint(inpainted_frame, mask)
    return inpainted_frame


def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # 상자가 너무 작은 것을 방지하기 위해 10픽셀 확대
            x1 = xmin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if x1 < 0:
                x1 = 0
            y1 = ymin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if y1 < 0:
                y1 = 0
            x2 = xmax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            y2 = ymax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), (255, 255, 255), thickness=-1)
    return mask


def inpaint_video(video_path, sub_list):
    index = 0
    frame_to_inpaint_list = []
    video_cap = cv2.VideoCapture(video_path)
    while True:
        # 비디오 프레임 읽기
        ret, frame = video_cap.read()
        if not ret:
            break
        index += 1
        if index in sub_list.keys():
            frame_to_inpaint_list.append((index, frame, sub_list[index]))
        if len(frame_to_inpaint_list) > config.PROPAINTER_MAX_LOAD_NUM:
            batch_results = parallel_inference(frame_to_inpaint_list)
            for index, frame in batch_results:
                file_name = f'/home/yao/Documents/Project/video-subtitle-remover/test/temp/{index}.png'
                cv2.imwrite(file_name, frame)
                print(f"success write: {file_name}")
            frame_to_inpaint_list.clear()
    print(f'finished')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
