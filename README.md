# Blot Profiler

웨스턴 블롯 TIF 이미지에서 밴드를 자동으로 찾아 intensity를 정량하고, internal control(loading control)로
정규화한 뒤 그래프까지 그려주는 macOS용 GUI 프로그램입니다. Python(PySide6 + NumPy/SciPy/scikit-image/matplotlib)으로 만들었습니다.

## 주요 기능

- **TIF 입력**: 8/16/32-bit 그레이스케일·RGB TIF (PNG/JPG도 가능). 어두운 밴드/밝은 배경(ECL·필름)은 자동 반전.
- **밴드 자동 검출**: 노이즈 기반 임계값 → 붙어 있는 밴드는 레인 간격으로 분리 → 가로줄(행)과 레인으로 자동 정리.
  밴드가 없는 레인(예: DMSO 레인의 Kras)에도 같은 위치에 박스를 만들어 0에 가까운 값으로 정량합니다.
- **같은 가로줄 = 같은 박스 크기**: 한 행(=한 단백질)의 모든 박스는 폭·높이가 같고, 행 표에서 한 번에 바꿉니다.
- **배경(background noise) 제거** 4가지: 없음 / Rolling ball(이미지 전체) / 박스 위·아래 로컬 배경 / 행 내 레인 사이 배경.
- **Target / Control 지정**: 각 행의 역할을 측정(Target), Control(Loading), 제외 중에서 고릅니다.
  Target ÷ Control을 레인별로 계산하고 기준 레인을 1로 정규화합니다.
- **사용자 그룹**: 이미지에서 밴드를 골라 그룹으로 묶습니다 (예: Erk 이중 밴드 합치기, 처리군 묶기).
- **수식 로직**: 행·그룹·밴드를 `+ - * / **`와 괄호로 조합합니다. 예) `(pErk_upper + pErk_lower) / Vinculin`, `G_IgG / Vinculin`, `R2L3 - R2L1`.
- **반복 실험 묶음**: 레인에 같은 '조건' 이름을 주면 그래프에서 평균 ± SD와 개별 점으로 표시합니다.
- **내보내기**: 밴드별 raw/배경/net 값과 결과 CSV, 그래프 PNG/PDF/SVG, 박스가 표시된 이미지, 분석 파일(.json, 다시 열어 수정 가능).

## 사용 순서

1. **1. 검출** 탭에서 TIF를 엽니다 (창에 끌어다 놓아도 됨). 자동으로 밴드를 찾습니다.
   흐린 밴드를 놓치면 *민감도*를 올리고, 마커(래더)가 잡히면 *좌우 가장자리 제외*를 늘린 뒤 다시 검출하세요.
2. 이미지에서 박스를 고칩니다.
   - 드래그: 이동 / 빈 곳 드래그: 여러 개 선택 / 방향키: 1px 이동(Shift: 5px)
   - 더블클릭: 그 위치에 밴드 추가 (가장 가까운 행·레인에 붙음) / Delete: 삭제
   - 우클릭: 그룹 추가·제거, 행 역할 지정, 선택한 밴드로 새 행 만들기
   - ⌘(Ctrl)+휠: 확대·축소, ⌘0: 창에 맞추기
3. **2. 행 / 레인** 탭에서 행 이름(단백질)과 역할을 지정하고, 박스 폭·높이를 조절합니다.
   레인 이름과 조건(반복 묶음)을 입력합니다. *레인 이름 일괄 입력*으로 한 번에 붙여 넣을 수 있습니다.
4. **3. 밴드 / 그룹** 탭에서 밴드별 정량값과 프로파일을 확인하고 그룹을 만듭니다.
5. **4. 정량 / 그래프** 탭에서 기준 레인을 고르고, 필요하면 수식을 추가합니다. CSV와 그래프를 저장합니다.

### 수식에서 쓸 수 있는 이름

| 이름 | 의미 |
| --- | --- |
| `Vinculin`, `pErk` … (행 이름, 공백·기호는 `_`로) 또는 `R1`, `R2` … | 그 행의 레인별 값 (벡터) |
| `R2L3` | 2번째 행, 3번째 레인 밴드 하나 |
| `그룹이름` | 그룹에 속한 밴드 값의 총합 (숫자 하나) |
| `G_그룹이름` | 그룹에 속한 밴드를 레인별로 합한 값 (벡터) |
| `sum mean max min log2 log10 sqrt` | 함수 |

벡터끼리의 연산은 레인별로 계산됩니다. 0으로 나누면 빈 값(NaN)이 됩니다.

### 정량 방식

- 정량값(Net) = 박스 안 픽셀 intensity 합(integrated density) − 배경.
- Rolling ball은 이미지 전체에서 부드러운 배경을 추정해 뺍니다 (반지름은 밴드 높이의 3~5배 이상 권장).
- 로컬 배경은 박스 바로 위·아래 띠의 중앙값을, 행 배경은 같은 행에서 박스 사이(레인 사이) 픽셀의 중앙값을 픽셀당 배경으로 뺍니다.
- 가장 정확한 결과를 위해 이미저에서 내보낸 **원본 16-bit TIF**를 쓰고, 포화(saturation)된 밴드는 피하세요.
  스크린샷·JPEG는 대비 조정과 압축 때문에 값이 왜곡될 수 있습니다.

## 설치 (macOS)

### .dmg로 설치

`BlotProfiler-<버전>.dmg`를 열고 **Blot Profiler.app**을 Applications 폴더로 끌어 놓으세요.
서명되지 않은 앱이라 처음 실행 시 경고가 나오면 Finder에서 앱을 우클릭 → *열기*를 누르세요.

### 직접 빌드

```bash
git clone https://github.com/windowrainYOON/blot-profiler.git
cd blot-profiler
./build_mac.sh            # dist/BlotProfiler-<버전>.dmg 생성
./build_mac.sh --install  # 빌드 후 /Applications 에 설치까지
```

Python 3.10 이상이 필요합니다 (`PYTHON=/path/to/python3 ./build_mac.sh` 로 지정 가능).

### 소스에서 바로 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_blot_profiler.py [이미지.tif]
```

## 개발

```bash
pip install -r requirements.txt pytest
python -m pytest tests
```

- `blot_profiler/core.py`: 이미지 로딩, 배경 제거, 밴드 검출, 정량, 수식 계산 (GUI와 독립)
- `blot_profiler/gui.py`: PySide6 GUI
- `packaging/`: PyInstaller 설정과 아이콘
- `tests/data/example_blot*.tif`: 예시 블롯 (스크린샷에서 잘라낸 저해상도 이미지, 테스트용)
