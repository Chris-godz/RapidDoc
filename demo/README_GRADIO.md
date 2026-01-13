# RapidDoc Gradio UI

간단한 웹 인터페이스로 RapidDoc의 문서 파싱 기능을 사용할 수 있습니다.

## 설치

### 1. Gradio 설치

```bash
# 가상환경 활성화
source venv/bin/activate

# Gradio 설치
pip install gradio
```

### 2. 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# Gradio UI 실행
python demo/gradio_app.py
```

실행 후 브라우저에서 자동으로 열립니다:
- **Local URL**: http://127.0.0.1:7860
- **Network URL**: http://0.0.0.0:7860

## 주요 기능

### 📁 파일 업로드
- PDF 파일 (.pdf)
- 이미지 파일 (.png, .jpg, .jpeg, .bmp, .tiff, .tif)

### ⚙️ 설정 옵션

#### 파싱 방법
- **auto**: 자동으로 최적의 방법 선택 (권장)
- **ocr**: OCR 강제 사용
- **txt**: 텍스트 추출만 (이미지 없는 PDF)

#### 기능 활성화
- **수식 인식**: LaTeX 형식의 수식 추출
- **표 인식**: 표를 HTML/Markdown으로 변환

#### 페이지 범위
- **시작 페이지**: 파싱 시작 페이지 (0부터 시작)
- **종료 페이지**: 파싱 종료 페이지 (0=전체)

#### 엔진 선택
- **Layout 엔진**: 문서 레이아웃 분석
  - `dxengine`: 최고 성능 (권장)
  - `onnxruntime`: 범용성
  - `openvino`: Intel CPU 최적화

- **OCR 엔진**: 텍스트 인식
  - `dxengine`: 최고 성능 (권장)
  - `onnxruntime`: 범용성
  - `openvino`: Intel CPU 최적화
  - `torch`: GPU 지원
  - `paddle`: PaddlePaddle

- **Formula 엔진**: 수식 인식
  - `onnxruntime`: 안정성 (권장)
  - `dxengine`: 고성능
  - `openvino`: Intel CPU 최적화

- **Table 엔진**: 표 인식
  - `dxengine`: 최고 성능 (권장)
  - `onnxruntime`: 범용성
  - `torch`: GPU 지원

### 📊 결과 확인

#### Markdown 결과 탭
- 추출된 텍스트, 표, 수식을 Markdown 형식으로 표시
- 복사 버튼으로 간편하게 복사 가능

#### 레이아웃 시각화 탭
- 인식된 레이아웃 영역을 시각화한 PDF
- 다운로드하여 확인 가능

#### 파싱 정보
- 처리 시간, 페이지 수
- 사용된 엔진 정보
- 출력 디렉토리 경로

## 권장 설정

### 최고 성능 (DX Engine 사용)
```
Layout Engine:  dxengine
OCR Engine:     dxengine
Formula Engine: onnxruntime
Table Engine:   dxengine
```

### 안정성 우선
```
Layout Engine:  onnxruntime
OCR Engine:     onnxruntime
Formula Engine: onnxruntime
Table Engine:   onnxruntime
```

### 균형잡힌 설정
```
Layout Engine:  dxengine
OCR Engine:     dxengine
Formula Engine: onnxruntime
Table Engine:   dxengine
```

## 사용 예시

### 1. 기본 사용
1. PDF 파일 업로드
2. 기본 설정 유지 (auto, 수식/표 활성화)
3. "🚀 파싱 시작" 클릭
4. 결과 확인

### 2. 특정 페이지만 파싱
1. 파일 업로드
2. 시작 페이지: 5, 종료 페이지: 10
3. 파싱 시작

### 3. 텍스트만 추출
1. 파일 업로드
2. 파싱 방법: txt
3. 수식 인식: 비활성화
4. 표 인식: 비활성화
5. 파싱 시작

## 출력 파일

파싱 결과는 `demo/output-gradio/` 디렉토리에 저장됩니다:

```
demo/output-gradio/
└── filename/
    └── auto/
        ├── filename.md              # Markdown 결과
        ├── filename_layout.pdf      # 레이아웃 시각화
        ├── filename_middle.json     # 중간 결과 (디버깅용)
        └── images/                  # 추출된 이미지
            ├── page_0_img_0.png
            └── ...
```

## 문제 해결

### Gradio 설치 오류
```bash
pip install --upgrade gradio
```

### 포트 변경
`gradio_app.py` 파일의 마지막 부분 수정:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,  # 원하는 포트로 변경
    share=False,
)
```

### 메모리 부족
- 페이지 범위를 제한하여 처리
- 더 작은 파일로 테스트

## FastAPI vs Gradio

| 항목 | FastAPI (app_offline.py) | Gradio (gradio_app.py) |
|------|-------------------------|------------------------|
| 인터페이스 | REST API | 웹 UI |
| 사용 용도 | 프로그래밍 통합 | 대화형 데모 |
| 파일 업로드 | HTTP POST | 드래그 앤 드롭 |
| 결과 확인 | JSON 응답 | 실시간 UI |
| 배치 처리 | 지원 | 단일 파일 |

## 참고

- CLI 버전: [demo_offline.py](demo_offline.py)
- FastAPI 버전: [app_offline.py](app_offline.py)
- 메인 README: [README_API_OFFLINE.md](README_API_OFFLINE.md)
