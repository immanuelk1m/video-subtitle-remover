import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import paddle
# ×××××××××××××××××××× [수정 금지] 시작 ××××××××××××××××××××
paddle.disable_signal_handler()
logging.disable(logging.DEBUG)  # DEBUG 로그 출력 비활성화
logging.disable(logging.WARNING)  # WARNING 로그 출력 비활성화
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# 해당 경로에 모델 전체 파일이 있는지 확인하고, 없으면 작은 파일들을 병합하여 전체 파일 생성
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# ffmpeg 실행 파일 경로 지정
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# ffmpeg에 실행 권한 추가
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ×××××××××××××××××××× [수정 금지] 끝 ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    """
    이미지 인페인팅 알고리즘 열거형
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ×××××××××××××××××××× [수정 가능] 시작 ××××××××××××××××××××
# H.264 인코딩 사용 여부, 안드로이드 폰에서 생성된 비디오 공유 시 이 옵션 활성화 필요
USE_H264 = True

# ×××××××××× 공통 설정 시작 ××××××××××
"""
MODE 선택 가능 알고리즘 유형
- InpaintMode.STTN 알고리즘: 실제 인물 영상에 효과 좋음, 속도 빠름, 자막 감지 건너뛰기 가능
- InpaintMode.LAMA 알고리즘: 애니메이션 영상에 효과 좋음, 속도 보통, 자막 감지 건너뛰기 불가
- InpaintMode.PROPAINTER 알고리즘: 많은 GPU 메모리 소모, 속도 느림, 움직임이 매우 격렬한 영상에 효과 좋음
"""
# 【인페인트 알고리즘 설정】
MODE = InpaintMode.STTN
# 【픽셀 편차 설정】
# 비자막 영역인지 판단하는 데 사용 (일반적으로 자막 텍스트 상자는 너비가 높이보다 크다고 가정. 만약 높이가 너비보다 크고 그 차이가 지정된 픽셀 크기를 초과하면 잘못된 감지로 간주)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# 마스크 크기를 확대하여 자동 감지된 텍스트 상자가 너무 작아 인페인팅 단계에서 글자 테두리가 남는 것을 방지
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# 두 텍스트 상자가 동일한 줄의 자막인지 판단하는 데 사용, 높이 차이가 지정된 픽셀 이내이면 동일한 줄로 간주
THRESHOLD_HEIGHT_DIFFERENCE = 20
# 두 자막 텍스트의 사각형 상자가 유사한지 판단하는 데 사용, X축과 Y축 편차가 모두 지정된 임계값 내에 있으면 동일한 텍스트 상자로 간주
PIXEL_TOLERANCE_Y = 20  # 감지 상자의 세로 방향 허용 픽셀 편차
PIXEL_TOLERANCE_X = 20  # 감지 상자의 가로 방향 허용 픽셀 편차
# ×××××××××× 공통 설정 끝 ××××××××××

# ×××××××××× InpaintMode.STTN 알고리즘 설정 시작 ××××××××××
# 아래 파라미터는 STTN 알고리즘 사용 시에만 적용됩니다.
"""
1. STTN_SKIP_DETECTION
의미: 감지 건너뛰기 사용 여부
효과: True로 설정하면 자막 감지를 건너뛰어 시간을 크게 절약할 수 있지만, 자막 없는 비디오 프레임을 잘못 처리하거나 제거된 자막이 누락될 수 있습니다.

2. STTN_NEIGHBOR_STRIDE
의미: 인접 프레임 간격. 예를 들어 50번째 프레임의 누락 영역을 채워야 할 때 STTN_NEIGHBOR_STRIDE=5이면, 알고리즘은 45번째, 40번째 프레임 등을 참조로 사용합니다.
효과: 참조 프레임 선택의 밀도를 제어합니다. 간격이 크면 더 적고 분산된 참조 프레임을 사용하고, 간격이 작으면 더 많고 집중된 참조 프레임을 사용합니다.

3. STTN_REFERENCE_LENGTH
의미: 참조 프레임 수. STTN 알고리즘은 각 복원 대상 프레임의 앞뒤 여러 프레임을 확인하여 복원에 필요한 컨텍스트 정보를 얻습니다.
효과: 값을 높이면 GPU 메모리 사용량이 증가하고 처리 효과가 좋아지지만, 처리 속도가 느려집니다.

4. STTN_MAX_LOAD_NUM
의미: STTN 알고리즘이 한 번에 로드할 수 있는 최대 비디오 프레임 수
효과: 값을 높이면 속도가 느려지지만 효과는 좋아집니다.
주의: STTN_MAX_LOAD_NUM은 STTN_NEIGHBOR_STRIDE와 STTN_REFERENCE_LENGTH보다 커야 합니다.
"""
STTN_SKIP_DETECTION = True
# 참조 프레임 간격
STTN_NEIGHBOR_STRIDE = 5
# 참조 프레임 길이 (수량)
STTN_REFERENCE_LENGTH = 10
# STTN 알고리즘이 동시에 처리할 최대 프레임 수 설정
STTN_MAX_LOAD_NUM = 50
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN 알고리즘 설정 끝 ××××××××××

# ×××××××××× InpaintMode.PROPAINTER 알고리즘 설정 시작 ××××××××××
# 【자신의 GPU 메모리 크기에 맞게 설정】 동시에 처리할 최대 이미지 수. 값을 높이면 처리 효과가 좋아지지만 더 많은 메모리가 필요합니다.
# 1280x720p 비디오: 80 설정 시 25GB 필요, 50 설정 시 19GB 필요
# 720x480p 비디오: 80 설정 시 8GB 필요, 50 설정 시 7GB 필요
PROPAINTER_MAX_LOAD_NUM = 70
# ×××××××××× InpaintMode.PROPAINTER 알고리즘 설정 끝 ××××××××××

# ×××××××××× InpaintMode.LAMA 알고리즘 설정 시작 ××××××××××
# 초고속 모드 활성화 여부. 활성화 시 인페인트 효과는 보장되지 않으며, 텍스트가 포함된 영역의 텍스트만 제거합니다.
LAMA_SUPER_FAST = False
# ×××××××××× InpaintMode.LAMA 알고리즘 설정 끝 ××××××××××
# ×××××××××××××××××××× [수정 가능] 끝 ××××××××××××××××××××
