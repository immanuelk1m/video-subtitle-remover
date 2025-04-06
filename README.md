# 비디오 자막 제거기 (Video Subtitle Remover)

비디오에서 자막을 자동으로 감지하고 제거하는 도구입니다.

## 주요 기능

- 비디오 파일에서 자막 영역 자동 감지
- 감지된 자막 영역 제거
- 다양한 비디오 포맷 지원
- 고품질 영상 처리

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU만으로도 동작 가능)

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/video-subtitle-remover.git
cd video-subtitle-remover
```

2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

1. 비디오 파일 준비
2. 다음 명령어로 실행:

```bash
python main.py --input [입력_비디오_경로] --output [출력_비디오_경로]
```

## 주요 의존성

- OpenCV
- PyTorch
- Albumentations
- 기타 이미지 처리 라이브러리

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 문의사항

버그 리포트 및 기능 제안은 GitHub Issues를 이용해 주세요.
