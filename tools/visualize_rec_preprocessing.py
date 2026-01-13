#!/usr/bin/env python3
"""
OCR Recognition 전처리 시각화 도구

이 스크립트는 OCR Recognition 모델의 전처리 과정을 단계별로 시각화합니다.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import math

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class RecPreprocessingVisualizer:
    """OCR Recognition 전처리 시각화기"""
    
    def __init__(self, input_height: int = 48, input_width: int = 640):
        """
        Args:
            input_height: 입력 이미지 높이 (기본값: 48)
            input_width: 입력 이미지 최대 너비 (기본값: 640)
        """
        self.rec_image_shape = [3, input_height, input_width]  # C, H, W
        self.mean = 0.5
        self.std = 0.5
    
    def preprocess_with_steps(
        self, 
        img: np.ndarray, 
        max_wh_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        전처리를 단계별로 수행하고 중간 결과 반환
        
        Args:
            img: 입력 이미지 (H, W, C), BGR, uint8
            max_wh_ratio: 최대 가로/세로 비율
            
        Returns:
            (최종 전처리 이미지, 중간 단계 딕셔너리)
        """
        imgC, imgH, imgW_base = self.rec_image_shape
        
        steps = {}
        
        # 원본 이미지 저장
        steps['0_original'] = {
            'image': img.copy(),
            'shape': img.shape,
            'dtype': img.dtype,
            'range': (img.min(), img.max()),
            'description': '원본 이미지 (H, W, C), BGR, uint8'
        }
        
        # [1] 동적 너비 계산
        if max_wh_ratio is None:
            h, w = img.shape[:2]
            max_wh_ratio = max(imgW_base / imgH, w / float(h))
        
        imgW_dynamic = int(imgH * max_wh_ratio)
        
        steps['1_dynamic_width'] = {
            'max_wh_ratio': max_wh_ratio,
            'imgW_base': imgW_base,
            'imgW_dynamic': imgW_dynamic,
            'description': f'동적 너비 계산: {imgW_base}px → {imgW_dynamic}px (ratio={max_wh_ratio:.2f})'
        }
        
        # [2] Resize
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW_dynamic:
            resized_w = imgW_dynamic
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        
        steps['2_resized'] = {
            'image': resized_image.copy(),
            'shape': resized_image.shape,
            'dtype': resized_image.dtype,
            'original_size': (h, w),
            'resized_size': (imgH, resized_w),
            'ratio': ratio,
            'description': f'Resize: ({h}, {w}) → ({imgH}, {resized_w}), ratio={ratio:.2f}'
        }
        
        # [3] Transpose
        transposed = resized_image.transpose((2, 0, 1))  # HWC → CHW
        
        steps['3_transposed'] = {
            'shape': transposed.shape,
            'description': f'Transpose: (H, W, C) → (C, H, W) = {transposed.shape}'
        }
        
        # [4] Normalize
        normalized = transposed.astype(np.float32) / 255.0
        
        steps['4_normalized'] = {
            'image_channel0': normalized[0].copy(),  # 시각화용 첫 번째 채널
            'shape': normalized.shape,
            'dtype': normalized.dtype,
            'range': (normalized.min(), normalized.max()),
            'description': f'Normalize: [0, 255] → [0, 1], range=({normalized.min():.3f}, {normalized.max():.3f})'
        }
        
        # [5] Standardize
        standardized = (normalized - self.mean) / self.std
        
        steps['5_standardized'] = {
            'image_channel0': standardized[0].copy(),  # 시각화용 첫 번째 채널
            'shape': standardized.shape,
            'dtype': standardized.dtype,
            'range': (standardized.min(), standardized.max()),
            'description': f'Standardize: (x - {self.mean}) / {self.std} → [-1, 1], range=({standardized.min():.3f}, {standardized.max():.3f})'
        }
        
        # [6] Padding
        padding_im = np.zeros((imgC, imgH, imgW_dynamic), dtype=np.float32)
        padding_im[:, :, :resized_w] = standardized
        
        steps['6_padded'] = {
            'image_channel0': padding_im[0].copy(),  # 시각화용 첫 번째 채널
            'shape': padding_im.shape,
            'dtype': padding_im.dtype,
            'range': (padding_im.min(), padding_im.max()),
            'padding_width': imgW_dynamic - resized_w,
            'description': f'Padding: ({imgC}, {imgH}, {resized_w}) → ({imgC}, {imgH}, {imgW_dynamic}), 패딩={imgW_dynamic - resized_w}px'
        }
        
        return padding_im, steps
    
    def visualize_steps(self, img: np.ndarray, output_path: Optional[str] = None):
        """
        전처리 단계를 시각화하여 저장
        
        Args:
            img: 입력 이미지 (H, W, C), BGR, uint8
            output_path: 출력 이미지 경로 (None이면 표시만)
        """
        final_image, steps = self.preprocess_with_steps(img)
        
        # 시각화 생성
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('OCR Recognition Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        # 1. 원본 이미지
        ax1 = plt.subplot(3, 3, 1)
        original = steps['0_original']['image']
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"[0] 원본 이미지\n{steps['0_original']['shape']}", fontsize=10)
        ax1.axis('off')
        
        # 2. 동적 너비 계산 (텍스트로 표시)
        ax2 = plt.subplot(3, 3, 2)
        ax2.axis('off')
        dw = steps['1_dynamic_width']
        info_text = (
            f"[1] 동적 너비 계산\n\n"
            f"max_wh_ratio = {dw['max_wh_ratio']:.2f}\n"
            f"imgW_base = {dw['imgW_base']}px\n"
            f"imgW_dynamic = {dw['imgW_dynamic']}px\n\n"
            f"→ {dw['imgW_base']}px → {dw['imgW_dynamic']}px"
        )
        ax2.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Resize
        ax3 = plt.subplot(3, 3, 3)
        resized = steps['2_resized']['image']
        ax3.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        ax3.set_title(f"[2] Resize\n{steps['2_resized']['resized_size']}", fontsize=10)
        ax3.axis('off')
        
        # 4. Transpose (텍스트로 표시)
        ax4 = plt.subplot(3, 3, 4)
        ax4.axis('off')
        trans_text = (
            f"[3] Transpose\n\n"
            f"(H, W, C) → (C, H, W)\n"
            f"{steps['2_resized']['shape']} → {steps['3_transposed']['shape']}"
        )
        ax4.text(0.5, 0.5, trans_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 5. Normalize
        ax5 = plt.subplot(3, 3, 5)
        norm_img = steps['4_normalized']['image_channel0']
        im5 = ax5.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
        ax5.set_title(f"[4] Normalize (Channel 0)\nrange: {steps['4_normalized']['range']}", fontsize=10)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 6. Standardize
        ax6 = plt.subplot(3, 3, 6)
        std_img = steps['5_standardized']['image_channel0']
        im6 = ax6.imshow(std_img, cmap='gray', vmin=-1, vmax=1)
        ax6.set_title(f"[5] Standardize (Channel 0)\nrange: {steps['5_standardized']['range']}", fontsize=10)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 7. Padding
        ax7 = plt.subplot(3, 3, 7)
        pad_img = steps['6_padded']['image_channel0']
        im7 = ax7.imshow(pad_img, cmap='gray', vmin=-1, vmax=1)
        ax7.set_title(f"[6] Padding (Channel 0)\n패딩={steps['6_padded']['padding_width']}px", fontsize=10)
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        # 8. 최종 결과 (패딩 영역 강조)
        ax8 = plt.subplot(3, 3, 8)
        pad_img_highlight = pad_img.copy()
        resized_w = steps['6_padded']['shape'][2] - steps['6_padded']['padding_width']
        # 패딩 영역을 빨간색으로 표시 (시각화용)
        pad_highlight = np.zeros((pad_img_highlight.shape[0], pad_img_highlight.shape[1], 3))
        pad_highlight[:, :resized_w, :] = np.stack([pad_img_highlight[:, :resized_w]] * 3, axis=-1)
        pad_highlight[:, resized_w:, 0] = 0.5  # 패딩 영역을 빨간색으로
        ax8.imshow(pad_highlight, interpolation='nearest')
        ax8.set_title(f"패딩 영역 강조 (빨강)\n원본={resized_w}px, 패딩={steps['6_padded']['padding_width']}px", fontsize=10)
        ax8.axis('off')
        
        # 9. 통계 정보
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = (
            f"📊 최종 통계\n\n"
            f"입력: {steps['0_original']['shape']}\n"
            f"출력: {steps['6_padded']['shape']}\n\n"
            f"dtype: {steps['6_padded']['dtype']}\n"
            f"range: [{steps['6_padded']['range'][0]:.3f}, {steps['6_padded']['range'][1]:.3f}]\n\n"
            f"메모리:\n"
            f"  원본: {original.nbytes / 1024:.1f} KB\n"
            f"  전처리: {final_image.nbytes / 1024:.1f} KB"
        )
        ax9.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                family='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ 시각화 저장: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return steps


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR Recognition 전처리 시각화')
    parser.add_argument('--image', type=str, required=True, help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 이미지 경로 (기본: 입력파일명_preprocessing.png)')
    parser.add_argument('--height', type=int, default=48, help='입력 높이 (기본: 48)')
    parser.add_argument('--width', type=int, default=640, help='입력 최대 너비 (기본: 640)')
    
    args = parser.parse_args()
    
    # 입력 이미지 로드
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {args.image}")
        return
    
    print(f"✅ 이미지 로드: {args.image}")
    print(f"   원본 크기: {img.shape}")
    
    # 출력 경로 설정
    if args.output is None:
        input_path = Path(args.image)
        args.output = str(input_path.parent / f"{input_path.stem}_preprocessing.png")
    
    # 시각화 생성
    visualizer = RecPreprocessingVisualizer(
        input_height=args.height,
        input_width=args.width
    )
    
    print(f"\n📊 전처리 파이프라인 시각화 중...")
    steps = visualizer.visualize_steps(img, args.output)
    
    # 단계별 정보 출력
    print("\n" + "=" * 80)
    print("📋 전처리 단계별 상세 정보")
    print("=" * 80)
    
    for key, value in steps.items():
        if 'description' in value:
            print(f"\n{value['description']}")
            if 'shape' in value:
                print(f"  Shape: {value['shape']}")
            if 'range' in value:
                print(f"  Range: {value['range']}")
    
    print("\n" + "=" * 80)
    print(f"✅ 완료! 시각화 이미지: {args.output}")


if __name__ == '__main__':
    main()
