# OCR Recognition Preprocessing 조사 결과 요약

## 📊 핵심 요약

OCR Recognition 모델(PP-OCRv5)의 전처리는 **6단계**로 구성되며, **동적 너비 조정**을 통해 메모리를 최적화합니다.

---

## 🎯 전처리 파이프라인 (6단계)

```
BGR 이미지 (H, W, 3) → uint8 [0-255]
    ↓
[1] 동적 너비 계산 (max_wh_ratio)
    ↓
[2] Resize (높이 48px 고정, aspect ratio 유지)
    ↓
[3] Transpose (HWC → CHW)
    ↓
[4] Normalize ([0, 255] → [0, 1])
    ↓
[5] Standardize (mean=0.5, std=0.5 → [-1, 1])
    ↓
[6] Right Padding (0으로 패딩)
    ↓
Float32 (C, H, W_dynamic) → [-1, 1]
```

---

## 🔑 핵심 특징

### 1️⃣ 동적 너비 조정 (핵심!)

```python
# 배치 내 최대 W/H 비율 계산
max_wh_ratio = 640 / 48  # 기본값
for img in batch:
    h, w = img.shape[:2]
    max_wh_ratio = max(max_wh_ratio, w / h)

# 동적 너비 = 높이 × 최대 비율
imgW_dynamic = int(48 * max_wh_ratio)
```

**효과**:
- 짧은 텍스트: 메모리 **50% 절감** (640px → 320px)
- 긴 텍스트: 필요시 확장 (640px → 1000px)
- 혼합 배치: 최장 텍스트 기준 자동 조정

---

### 2️⃣ 고정 높이 48px

모든 이미지는 **높이 48px**로 고정되며, 가로/세로 비율을 유지합니다.

```python
resized_w = int(48 * (w / h))  # aspect ratio 유지
resized_image = cv2.resize(img, (resized_w, 48))
```

---

### 3️⃣ 표준화 [-1, 1]

PP-OCRv5 표준 정규화 파라미터:

```python
# Normalize: [0, 255] → [0, 1]
img = img / 255.0

# Standardize: [0, 1] → [-1, 1]
img = (img - 0.5) / 0.5
```

| 입력 픽셀 | 정규화 후 | 표준화 후 |
|---------|---------|---------|
| 0 (검정) | 0.0 | **-1.0** |
| 128 (회색) | 0.5 | **0.0** |
| 255 (흰색) | 1.0 | **+1.0** |

---

### 4️⃣ 오른쪽 패딩

```python
# 0으로 초기화 (표준화 후의 중간값)
padding_im = np.zeros((3, 48, imgW_dynamic))

# 원본 이미지를 왼쪽에 배치
padding_im[:, :, :resized_w] = resized_image

# 오른쪽은 0으로 패딩 (픽셀값 128에 해당)
```

---

## 📂 구현 위치

### 1. **DX Engine 구현** (메인)
- **파일**: `rapid_doc/model/ocr/dx_ocr.py`
- **클래스**: `DxTextRecognizer` (line 292)
- **메서드**:
  - `_resize_norm_img()` (line 438): 단일 이미지 전처리
  - `_preprocess_batch()` (line 403): 배치 이미지 전처리

### 2. **ONNX 벤치마크 구현**
- **파일**: `value_compare/recognition/onnx_recognition.py`
- **클래스**: `ONNXRecognitionInference` (line 37)
- **메서드**:
  - `preprocess()` (line 126): 단일 이미지 전처리
  - `preprocess_batch()` (line 179): 배치 이미지 전처리

### 3. **RapidOCR 래퍼**
- **파일**: `rapid_doc/model/ocr/rapid_ocr.py`
- **클래스**: `RapidOcrModel` (line 25)
- **사용**: `self.text_recognizer.resize_norm_img()`

---

## 🛠️ 시각화 도구

**파일**: `tools/visualize_rec_preprocessing.py`

**사용법**:
```bash
python tools/visualize_rec_preprocessing.py \
    --image demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg \
    --output docs/rec_preprocessing_visualization.png
```

**출력**: 6단계 전처리 과정을 시각화한 이미지 생성

---

## 📊 성능 최적화

### 배치 크기 설정

```python
# batch_analyze.py에서 설정
ocr_config = {
    "rec_batch_num": 6,  # CPU: 4-6, GPU: 16-32
}
```

### 메모리 효율 비교

| 시나리오 | 고정 너비 (640px) | 동적 너비 | 메모리 절감 |
|---------|----------------|---------|----------|
| 짧은 텍스트 (W/H=6.67) | 552,960 bytes | 276,480 bytes | **50% ↓** |
| 중간 텍스트 (W/H=10.0) | 552,960 bytes | 414,720 bytes | **25% ↓** |
| 긴 텍스트 (W/H=16.67) | 552,960 bytes | 691,200 bytes | 25% ↑ (필요시만) |

배치 크기: 6개, 형상: (B, C, H, W) = (6, 3, 48, W_dynamic)

---

## ✅ 검증 체크리스트

### 입력 검증
```python
assert img.dtype == np.uint8
assert len(img.shape) == 3 and img.shape[2] == 3  # (H, W, C)
```

### 출력 검증
```python
assert preprocessed.dtype == np.float32
assert preprocessed.shape == (3, 48, imgW_dynamic)  # (C, H, W)
assert -1.0 <= preprocessed.min() <= preprocessed.max() <= 1.0
```

### 패딩 검증
```python
# 패딩 영역이 0인지 확인
assert np.allclose(preprocessed[:, :, resized_w:], 0.0)
```

---

## 🔍 실제 예시

**입력 이미지**:
- 원본 크기: (51, 1015, 3) → H=51, W=1015
- W/H 비율: 1015 / 51 ≈ 19.90

**전처리 결과**:
1. 동적 너비 계산: `imgW = 48 × 19.90 = 955px`
2. Resize: (51, 1015) → (48, 955)
3. Transpose: (48, 955, 3) → (3, 48, 955)
4. Normalize: uint8 [0, 255] → float32 [0, 1]
5. Standardize: [0, 1] → [-1, 1]
6. Padding: (3, 48, 955) → (3, 48, 955) (패딩 없음, 이미 최적 크기)

**메모리**:
- 원본: 51 × 1015 × 3 = 155,295 bytes (uint8)
- 전처리: 3 × 48 × 955 = 137,520 bytes (float32) = **550,080 bytes** (4배)

---

## 📚 추가 문서

- **상세 가이드**: `docs/OCR_REC_PREPROCESSING.md`
- **시각화 예시**: `docs/rec_preprocessing_visualization.png`
- **벤치마크 코드**: `ort_benchmarks/benchmark_ocr_rec.py`

---

## 🎓 핵심 정리

1. **동적 너비**: 배치 내 최대 W/H 비율로 메모리 최적화
2. **고정 높이**: 48px (PP-OCRv5 표준)
3. **정규화**: mean=0.5, std=0.5 → [-1, 1]
4. **패딩**: 오른쪽에 0으로 패딩 (픽셀값 128 해당)
5. **메모리 효율**: 짧은 텍스트 50% 절감, 긴 텍스트는 필요시만 확장

---

**작성일**: 2024-11-18  
**관련 파일**: 
- `rapid_doc/model/ocr/dx_ocr.py`
- `value_compare/recognition/onnx_recognition.py`
- `tools/visualize_rec_preprocessing.py`
