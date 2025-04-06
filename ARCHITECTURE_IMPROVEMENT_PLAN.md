# Video Subtitle Remover 아키텍처 개선 계획

## 1. 목표

기존의 GUI 기반 애플리케이션에서 GUI 부분을 제거하고, 사용자가 터미널에서 명령줄 인자를 통해 자막 제거 기능을 사용할 수 있도록 명령줄 인터페이스(CLI)를 구현합니다. 이를 통해 프로젝트의 사용 방식을 변경하고, GUI 관련 의존성을 제거합니다.

## 2. 세부 개선 계획

### 2.1. GUI 코드 제거

*   **`gui.py` 파일 삭제:** 프로젝트 루트 디렉토리에 있는 `gui.py` 파일을 완전히 삭제합니다.
*   **관련 코드 정리:**
    *   `backend/main.py` 및 기타 파일에서 `gui.py` 또는 `PySimpleGUI` 관련 임포트 구문을 제거합니다.
    *   `SubtitleRemover` 클래스 내 `gui_mode` 플래그 및 관련 조건문 로직을 제거합니다.
    *   `preview_frame` 속성 및 관련 업데이트 로직을 제거합니다.
    *   GUI 진행률 표시와 관련된 로직(`progress_total` 업데이트 방식 등)을 CLI 환경에 맞게 조정하거나 제거합니다.

### 2.2. CLI 인터페이스 구현

*   **`argparse` 사용:** `backend/main.py` 파일의 메인 실행 블록(`if __name__ == '__main__':`)에서 파이썬 표준 라이브러리 `argparse`를 사용하여 명령줄 인자를 파싱하도록 구현합니다.
*   **명령줄 인자 정의:**
    *   `--input` (필수): 처리할 입력 비디오 또는 이미지 파일 경로.
    *   `--output` (필수): 결과 파일을 저장할 경로.
    *   `--sub_area` (선택): 자막 영역을 지정하는 좌표 (형식: `ymin,ymax,xmin,xmax`). 지정하지 않으면 전체 영역 또는 자동 감지 사용.
    *   `--mode` (선택): 사용할 인페인팅 모드 (`lama`, `sttn`, `propainter`). 기본값 설정 필요 (예: `lama`).
    *   기타 필요한 설정값(예: 모델 경로, 임계값 등)을 인자로 추가하거나, 기존 `config.py`를 계속 활용합니다.
*   **기존 입력 방식 제거:** `backend/main.py`의 `input()` 함수를 사용한 사용자 입력 로직(line 914 부근)을 제거합니다.
*   **진행 상황 출력:** `tqdm` 라이브러리 또는 `print` 문을 사용하여 파일 처리 진행 상황을 터미널에 명확하게 표시합니다.

### 2.3. 의존성 정리

*   `requirements.txt` 파일이 존재한다면, `PySimpleGUI` 라이브러리 의존성을 제거하고 파일을 업데이트합니다.

## 3. 개선 후 예상 아키텍처

```mermaid
graph TD
    A[User (Terminal)] -- Command Line Arguments --> B(backend/main.py - CLI Interface);
    B -- Parse Arguments (argparse) --> C{Subtitle Remover Logic};
    C -- Use --> D[SubtitleDetect Module];
    C -- Use --> E[Inpaint Module];
    C -- Use --> F[FFmpeg/OpenCV Module];
    C -- Print Progress/Result --> A;
    subgraph Backend Modules
        C
        D
        E
        F
    end
```

## 4. 기대 효과

*   GUI 관련 코드 및 의존성 제거로 프로젝트 단순화.
*   스크립트나 자동화 환경에서 프로젝트를 쉽게 활용 가능.
*   명령줄 사용에 익숙한 사용자에게 더 편리한 인터페이스 제공.