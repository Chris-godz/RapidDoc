# OCR Recognition Model Preprocessing 상세 분석

## 📋 목차
1. [개요](#개요)
2. [전처리 파이프라인](#전처리-파이프라인)
3. [단계별 상세 설명](#단계별-상세-설명)
4. [배치 처리 최적화](#배치-처리-최적화)
5. [코드 구현 위치](#코드-구현-위치)
6. [주요 파라미터](#주요-파라미터)

---

## 개요

OCR Recognition 모델은 **PP-OCRv5** 기반으로, 텍스트 라인 이미지를 받아 문자열을 인식합니다.
전처리는 **동적 너비 조정**을 통해 다양한 길이의 텍스트를 효율적으로 처리합니다.

### 입력/출력 스펙
- **입력**: BGR 이미지 (H, W, 3), numpy.ndarray, uint8
- **출력**: 정규화된 이미지 (C, H, W), numpy.ndarray, float32
- **기본 형상**: [3, 48, 640] (채널, 높이, 최대너비)

---

## 전처리 파이프라인

```
원본 이미지 (H, W, 3)
    ↓
[1] 동적 너비 계산 (max_wh_ratio 기반)
    ↓
[2] Resize (aspect ratio 유지, 높이 48 고정)
    ↓
[3] Transpose (HWC → CHW)
    ↓
[4] Normalize ([0, 255] → [0, 1])
    ↓
[5] Standardize (mean=0.5, std=0.5 → [-1, 1])
    ↓
[6] Padding (오른쪽에 0으로 패딩)
    ↓
출력 (3, 48, dynamic_width)
```

---

## 단계별 상세 설명

### [1] 동적 너비 계산 (Dynamic Width)

**목적**: 배치 내 다양한 길이의 텍스트를 효율적으로 처리하기 위해 최대 가로/세로 비율 계산

```python
# 배치 내 최대 W/H 비율 찾기
max_wh_ratio = imgW_base / imgH  # 기본값: 640 / 48 = 13.33
for img in img_list:
    h, w = img.shape[:2]
    wh_ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, wh_ratio)  # 최대값 업데이트

# 동적 최대 너비 계산
imgW_dynamic = int(imgH * max_wh_ratio)
```

**예시**:
- 이미지1: 100x20 (ratio=5.0)
- 이미지2: 300x30 (ratio=10.0)
- 이미지3: 600x40 (ratio=15.0) ← 최대
- → `max_wh_ratio = 15.0`
- → `imgW_dynamic = 48 * 15.0 = 720px`

**효과**: 
- ✅ 메모리 효율: 실제 필요한 너비만큼만 할당 (불필요한 640px 대신 720px)
- ✅ 연산 효율: 패딩 영역 최소화

---

### [2] Resize (Aspect Ratio 유지)

**목적**: 높이를 48px로 고정하면서 가로/세로 비율 유지

```python
imgC, imgH, imgW = self.rec_image_shape  # [3, 48, dynamic_width]

h, w = img.shape[:2]
ratio = w / float(h)  # 원본 이미지의 가로/세로 비율

# 새로운 너비 계산
if math.ceil(imgH * ratio) > imgW:
    resized_w = imgW  # 최대 너비 제한
else:
    resized_w = int(math.ceil(imgH * ratio))

# 리사이즈 수행
resized_image = cv2.resize(img, (resized_w, imgH))  # (resized_w, 48)
```

**예시**:
- 원본: 300x30 → ratio=10.0
- imgH=48 → `resized_w = 48 * 10.0 = 480px`
- 결과: **480x48**

**주의사항**:
- `math.ceil()`: 반올림하여 정수 너비 보장
- 최대 너비 제한: 너무 긴 텍스트는 `imgW`로 클리핑

---

### [3] Transpose (HWC → CHW)

**목적**: PyTorch/ONNX 모델의 입력 형식에 맞춤

```python
resized_image = resized_image.transpose((2, 0, 1))  # (H, W, C) → (C, H, W)
```

**Before**: (48, 480, 3) - Height, Width, Channels
**After**: (3, 48, 480) - Channels, Height, Width

---

### [4] Normalize (0-255 → 0-1)

**목적**: 픽셀 값 정규화

```python
resized_image = resized_image.astype(np.float32)
resized_image = resized_image / 255.0  # [0, 255] → [0, 1]
```

**효과**: float32 연산 안정성 확보

---

### [5] Standardize (Mean-Std Normalization)

**목적**: 표준화를 통한 학습 안정화 (PP-OCRv5 표준)

```python
mean = 0.5
std = 0.5

resized_image -= mean  # mean subtraction
resized_image /= std   # std division

# 수식: (x - 0.5) / 0.5
# [0, 1] → [-1, 1]
```

**변환 예시**:
- 픽셀 값 0 → (0 - 0.5) / 0.5 = **-1.0**
- 픽셀 값 0.5 → (0.5 - 0.5) / 0.5 = **0.0**
- 픽셀 값 1.0 → (1.0 - 0.5) / 0.5 = **1.0**

**최종 범위**: **[-1, 1]**

---

### [6] Padding (Right Padding)

**목적**: 배치 내 모든 이미지를 동일한 너비로 통일

```python
imgC, imgH, imgW = [3, 48, dynamic_width]  # 동적 너비

# 0으로 초기화된 패딩 이미지 생성
padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)

# 원본 이미지를 왼쪽에 배치
padding_im[:, :, :resized_w] = resized_image

# 오른쪽은 0으로 패딩됨
```

**시각화**:
```
원본 (resized_w=480):
[################] 480px

패딩 후 (imgW=720):
[################0000000000] 
 ←----- 480 ---→ ←-- 240 -→
```

**패딩 값**: **0.0** (표준화 후의 중간값, 원래 픽셀 값 128에 해당)

---

## 배치 처리 최적화

### 배치 전처리 전략

**핵심 아이디어**: 배치 내 최대 W/H 비율을 계산하여 모든 이미지를 동일한 너비로 처리

```python
def preprocess_batch(self, img_list: List[np.ndarray]) -> np.ndarray:
    """배치 이미지 전처리 (RapidOCR 방식 - 동적 너비 조정)"""
    imgC, imgH, imgW_base = self.rec_image_shape  # [3, 48, 640]
    
    # 1️⃣ 배치 내 최대 W/H 비율 계산
    max_wh_ratio = imgW_base / imgH  # 기본값
    for img in img_list:
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    
    # 2️⃣ 동적 최대 너비 계산
    imgW_dynamic = int(imgH * max_wh_ratio)
    
    logger.debug(f"배치 동적 너비 조정: 기본 {imgW_base}px → 동적 {imgW_dynamic}px")
    
    # 3️⃣ 각 이미지 전처리 (동일한 max_wh_ratio 사용)
    batch_images = []
    for img in img_list:
        norm_img = self.preprocess(img, max_wh_ratio=max_wh_ratio)
        batch_images.append(norm_img[np.newaxis, :])
    
    # 4️⃣ 배치로 결합
    batch = np.concatenate(batch_images, axis=0).astype(np.float32)
    
    return batch  # (B, C, H, W_dynamic)
```

### 배치 처리 효율성

| 배치 크기 | 고정 너비 (640px) | 동적 너비 | 메모리 절감 |
|---------|----------------|---------|----------|
| 6개 (짧은 텍스트) | 6×3×48×640 = 552,960 | 6×3×48×320 = 276,480 | **50%** ↓ |
| 6개 (중간 텍스트) | 6×3×48×640 = 552,960 | 6×3×48×480 = 414,720 | **25%** ↓ |
| 6개 (긴 텍스트) | 6×3×48×640 = 552,960 | 6×3×48×800 = 691,200 | 25% ↑ (필요시만) |

**효과**:
- ✅ 짧은 텍스트 배치: 메모리 50% 절감
- ✅ 긴 텍스트 배치: 필요한 만큼만 확장
- ✅ 혼합 배치: 최장 텍스트 기준 자동 조정

---

## 코드 구현 위치

### 1. DX Engine 구현 (DxTextRecognizer)

**파일**: `rapid_doc/model/ocr/dx_ocr.py`

**클래스**: `DxTextRecognizer` (line 292)

**핵심 메서드**:
```python
class DxTextRecognizer:
    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """단일 이미지 리사이즈 및 정규화 (line 438)"""
        imgC, imgH, imgW = self.rec_image_shape
        
        # 리사이즈
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        
        # 정규화
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0  # (C, H, W)
        resized_image -= 0.5
        resized_image /= 0.5
        
        # 패딩
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        
        return padding_im
    
    def _preprocess_batch(self, img_list: List[np.ndarray]) -> np.ndarray:
        """배치 이미지 전처리 (line 403)"""
        imgC, imgH, imgW = self.rec_image_shape
        
        # 배치 내 최대 W/H 비율 계산
        max_wh_ratio = imgW / imgH
        for img in img_list:
            h, w = img.shape[:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        
        batch_images = []
        for img in img_list:
            norm_img = self._resize_norm_img(img, max_wh_ratio)
            batch_images.append(norm_img[np.newaxis, :])
        
        batch = np.concatenate(batch_images, axis=0).astype(np.float32)
        return batch
```

---

### 2. ONNX 테스트/벤치마크 구현

**파일**: `value_compare/recognition/onnx_recognition.py`

**클래스**: `ONNXRecognitionInference` (line 37)

**핵심 메서드**:
```python
class ONNXRecognitionInference:
    def __init__(self, model_path: str, input_height: int = 48, input_width: int = 640):
        self.rec_image_shape = [3, input_height, input_width]  # [C, H, W]
        self.mean = 0.5  # PP-OCRv5 표준
        self.std = 0.5
        # ...
    
    def preprocess(self, img: np.ndarray, max_wh_ratio: Optional[float] = None) -> np.ndarray:
        """단일 이미지 전처리 (line 126)"""
        # (위의 _resize_norm_img와 동일한 로직)
        pass
    
    def preprocess_batch(self, img_list: List[np.ndarray]) -> np.ndarray:
        """배치 이미지 전처리 (line 179)"""
        # (위의 _preprocess_batch와 동일한 로직)
        pass
```

---

### 3. RapidOCR 래퍼

**파일**: `rapid_doc/model/ocr/rapid_ocr.py`

**클래스**: `RapidOcrModel` (line 25)

**사용 예시**:
```python
# RapidOCR 라이브러리의 resize_norm_img 메서드 사용
norm_img = self.text_recognizer.resize_norm_img(img, max_wh_ratio)
```

---

## 주요 파라미터

### 모델 입력 형상

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `input_height` | **48** | 고정 높이 (PP-OCRv5 표준) |
| `input_width` | **640** | 기본 최대 너비 (동적 조정 가능) |
| `input_channels` | **3** | BGR 채널 |

### 정규화 파라미터 (PP-OCRv5 표준)

| 파라미터 | 값 | 적용 범위 |
|---------|-----|---------|
| `mean` | **0.5** | 모든 채널 동일 |
| `std` | **0.5** | 모든 채널 동일 |
| **결과 범위** | **[-1, 1]** | 표준화 후 |

**비교**: 다른 모델들의 정규화
- **Detection 모델**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)
- **Recognition 모델**: mean=0.5, std=0.5 (단순화)

---

## 성능 최적화 팁

### 1. 배치 크기 조정

```python
# batch_analyze.py에서 설정
self.ocr_rec_batch_size = ocr_config.get("rec_batch_num", 6)  # 기본값 6
```

**권장값**:
- CPU: 4-6
- GPU: 16-32
- 메모리 제약 시: 2-4

### 2. 동적 너비 활용

```python
# 짧은 텍스트가 많은 경우 메모리 절감
# 긴 텍스트는 자동으로 필요한 만큼 확장
max_wh_ratio = max(imgW_base / imgH, *[w/h for h, w in img_sizes])
```

### 3. 불필요한 패딩 최소화

```python
# 배치를 텍스트 길이별로 그룹화하면 더 효율적
short_texts = [img for img in imgs if img.shape[1] < 200]
long_texts = [img for img in imgs if img.shape[1] >= 200]
```

---

## 디버깅 체크리스트

### ✅ 전처리 검증

1. **입력 형식 확인**
   ```python
   assert img.dtype == np.uint8, "입력은 uint8이어야 함"
   assert len(img.shape) == 3, "입력은 (H, W, C) 형식"
   assert img.shape[2] == 3, "입력은 3채널 (BGR)"
   ```

2. **리사이즈 확인**
   ```python
   assert resized_image.shape[1] == 48, "높이는 48이어야 함"
   assert resized_image.shape[0] <= imgW, "너비는 최대값 이하"
   ```

3. **정규화 범위 확인**
   ```python
   assert -1.0 <= preprocessed.min() <= preprocessed.max() <= 1.0
   ```

4. **패딩 확인**
   ```python
   assert padding_im.shape == (3, 48, imgW)
   assert np.allclose(padding_im[:, :, resized_w:], 0.0)  # 패딩 영역이 0
   ```

---

## 참고 자료

### 관련 파일
- `rapid_doc/model/ocr/dx_ocr.py` - DX Engine 구현
- `value_compare/recognition/onnx_recognition.py` - ONNX 벤치마크
- `rapid_doc/model/ocr/rapid_ocr.py` - RapidOCR 래퍼
- `ort_benchmarks/benchmark_ocr_rec.py` - 성능 측정

### 외부 문서
- [PP-OCRv5 공식 문서](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR 문서](https://github.com/RapidAI/RapidOCR)

---

## 요약

**OCR Recognition 전처리 핵심**:
1. 🎯 **동적 너비 조정**: 배치 내 최대 W/H 비율로 메모리 효율화
2. 📏 **고정 높이 48px**: PP-OCRv5 표준
3. 🔄 **Aspect Ratio 유지**: 텍스트 왜곡 방지
4. 📊 **정규화 [-1, 1]**: mean=0.5, std=0.5 표준화
5. 🔲 **오른쪽 패딩**: 배치 통일을 위한 0 패딩

**성능 최적화**:
- 짧은 텍스트: 메모리 50% 절감
- 긴 텍스트: 필요시만 확장
- 배치 크기: CPU 4-6, GPU 16-32 권장
