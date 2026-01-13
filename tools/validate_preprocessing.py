#!/usr/bin/env python3
"""
전처리 시각화 및 검증 스크립트

onnx_recognition.py의 전처리가 올바른지 확인합니다.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from loguru import logger
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from value_compare.recognition.onnx_recognition import ONNXRecognitionInference


def visualize_preprocessing():
    """전처리 시각화"""
    
    # 이미지 로드
    image_path = "demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg"
    img_bgr = cv2.imread(image_path)
    
    logger.info(f"원본 이미지: {img_bgr.shape}, dtype={img_bgr.dtype}, range=[{img_bgr.min()}, {img_bgr.max()}]")
    
    # 인식기 초기화
    recognizer = ONNXRecognitionInference(model_path="onnx_models/ch_PP-OCRv5_rec_server_infer.onnx")
    
    # 전처리
    preprocessed = recognizer.preprocess(img_bgr)
    logger.info(f"전처리 후: {preprocessed.shape}, dtype={preprocessed.dtype}, range=[{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('OCR Recognition Preprocessing Visualization', fontsize=16, fontweight='bold')
    
    # 1. 원본 이미지 (RGB)
    ax1 = axes[0, 0]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title(f'1. 원본 (RGB)\n{img_bgr.shape}')
    ax1.axis('off')
    
    # 2. 전처리 후 - 채널 0 (B)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(preprocessed[0], cmap='gray', vmin=-1, vmax=1)
    ax2.set_title(f'2. 전처리 - Channel 0 (B)\nrange=[{preprocessed[0].min():.2f}, {preprocessed[0].max():.2f}]')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. 전처리 후 - 채널 1 (G)
    ax3 = axes[0, 2]
    im3 = ax3.imshow(preprocessed[1], cmap='gray', vmin=-1, vmax=1)
    ax3.set_title(f'3. 전처리 - Channel 1 (G)\nrange=[{preprocessed[1].min():.2f}, {preprocessed[1].max():.2f}]')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. 전처리 후 - 채널 2 (R)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(preprocessed[2], cmap='gray', vmin=-1, vmax=1)
    ax4.set_title(f'4. 전처리 - Channel 2 (R)\nrange=[{preprocessed[2].min():.2f}, {preprocessed[2].max():.2f}]')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. 히스토그램
    ax5 = axes[1, 1]
    ax5.hist(preprocessed[0].flatten(), bins=50, alpha=0.5, label='Channel 0 (B)')
    ax5.hist(preprocessed[1].flatten(), bins=50, alpha=0.5, label='Channel 1 (G)')
    ax5.hist(preprocessed[2].flatten(), bins=50, alpha=0.5, label='Channel 2 (R)')
    ax5.set_title('5. 히스토그램')
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 통계
    ax6 = axes[1, 2]
    ax6.axis('off')
    stats_text = (
        f"📊 전처리 통계\n\n"
        f"원본:\n"
        f"  Shape: {img_bgr.shape}\n"
        f"  Range: [{img_bgr.min()}, {img_bgr.max()}]\n"
        f"  Mean: {img_bgr.mean():.1f}\n\n"
        f"전처리:\n"
        f"  Shape: {preprocessed.shape}\n"
        f"  Range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]\n"
        f"  Mean: {preprocessed.mean():.3f}\n"
        f"  Std: {preprocessed.std():.3f}\n\n"
        f"✅ 정상 범위: [-1, 1]\n"
        f"✅ 평균 ~0.0 예상\n"
        f"{'❌ 비정상!' if abs(preprocessed.mean()) > 0.1 else '✅ 정상'}"
    )
    ax6.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            family='monospace')
    
    plt.tight_layout()
    
    output_path = 'docs/preprocessing_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 시각화 저장: {output_path}")
    plt.close()
    
    # 🔍 문제 진단
    logger.info("\n" + "=" * 80)
    logger.info("전처리 검증")
    logger.info("=" * 80)
    
    # 1. 범위 확인
    if preprocessed.min() < -1.1 or preprocessed.max() > 1.1:
        logger.error(f"❌ 값 범위 이상: [{preprocessed.min():.3f}, {preprocessed.max():.3f}] (예상: [-1, 1])")
    else:
        logger.info(f"✅ 값 범위 정상: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # 2. 평균 확인
    if abs(preprocessed.mean()) > 0.2:
        logger.warning(f"⚠️  평균값 이상: {preprocessed.mean():.3f} (예상: ~0.0)")
    else:
        logger.info(f"✅ 평균값 정상: {preprocessed.mean():.3f}")
    
    # 3. 패딩 영역 확인
    resized_w = int((img_bgr.shape[1] / img_bgr.shape[0]) * 48)
    padding_area = preprocessed[:, :, resized_w:]
    
    if padding_area.size > 0:
        logger.info(f"\n패딩 영역: {padding_area.shape}, 값={padding_area.mean():.3f}")
        if not np.allclose(padding_area, 0.0):
            logger.warning(f"⚠️  패딩 영역이 0이 아님: {padding_area.mean():.3f}")
        else:
            logger.info(f"✅ 패딩 영역 정상 (0.0)")
    
    # 4. 원본 이미지 확인
    logger.info(f"\n원본 이미지 확인:")
    logger.info(f"  크기: {img_bgr.shape}")
    logger.info(f"  평균: {img_bgr.mean():.1f}")
    
    # 원본 이미지가 거의 검정색 (0에 가까움) 또는 거의 흰색 (255에 가까움)인지 확인
    if img_bgr.mean() < 50:
        logger.warning(f"⚠️  원본 이미지가 너무 어둡습니다: 평균={img_bgr.mean():.1f}")
    elif img_bgr.mean() > 200:
        logger.warning(f"⚠️  원본 이미지가 너무 밝습니다: 평균={img_bgr.mean():.1f}")
    else:
        logger.info(f"✅ 원본 이미지 밝기 정상: 평균={img_bgr.mean():.1f}")


if __name__ == '__main__':
    visualize_preprocessing()
