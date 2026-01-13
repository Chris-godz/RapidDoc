"""
두 Detection 결과의 차이 원인 분석 스크립트
"""

import sys
from pathlib import Path
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

def analyze_difference():
    """저장된 결과를 분석하여 차이의 원인을 파악"""
    
    output_dir = Path("value_compare/detect/output_compare")
    
    # 박스 로드
    boxes1 = np.load(output_dir / "table_90_dxnn_detect_boxes.npy")
    boxes2 = np.load(output_dir / "table_90_dx_ocr_boxes.npy")
    
    logger.info("=" * 80)
    logger.info("Detection 결과 차이 분석")
    logger.info("=" * 80)
    logger.info(f"dxnn_detect: {len(boxes1)}개 박스")
    logger.info(f"dx_ocr:      {len(boxes2)}개 박스")
    logger.info(f"차이:        {len(boxes2) - len(boxes1)}개 ({(len(boxes2) - len(boxes1)) / len(boxes1) * 100:.1f}% 더 많음)")
    logger.info("=" * 80)
    
    # 박스 크기 분석
    def get_box_sizes(boxes):
        """박스 크기 계산 (면적)"""
        sizes = []
        for box in boxes:
            # 사각형 면적 계산 (대각선 길이의 곱 / 2)
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[1] - box[2])
            area = width * height
            sizes.append(area)
        return np.array(sizes)
    
    sizes1 = get_box_sizes(boxes1)
    sizes2 = get_box_sizes(boxes2)
    
    logger.info("\n박스 크기 통계 (면적, pixels²):")
    logger.info(f"dxnn_detect:")
    logger.info(f"  - 최소: {sizes1.min():.1f}")
    logger.info(f"  - 최대: {sizes1.max():.1f}")
    logger.info(f"  - 평균: {sizes1.mean():.1f}")
    logger.info(f"  - 중앙: {np.median(sizes1):.1f}")
    
    logger.info(f"\ndx_ocr:")
    logger.info(f"  - 최소: {sizes2.min():.1f}")
    logger.info(f"  - 최대: {sizes2.max():.1f}")
    logger.info(f"  - 평균: {sizes2.mean():.1f}")
    logger.info(f"  - 중앙: {np.median(sizes2):.1f}")
    
    # 작은 박스 개수 비교
    thresholds = [10, 50, 100, 200, 500]
    
    logger.info("\n작은 박스 개수 (면적 기준):")
    logger.info(f"{'Threshold':<12} {'dxnn_detect':<15} {'dx_ocr':<15} {'차이':<10}")
    logger.info("-" * 60)
    
    for thresh in thresholds:
        count1 = (sizes1 < thresh).sum()
        count2 = (sizes2 < thresh).sum()
        diff = count2 - count1
        logger.info(f"< {thresh:>5} px²   {count1:<15} {count2:<15} +{diff}")
    
    # 히스토그램 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 박스 개수 비교
    axes[0, 0].bar(['dxnn_detect', 'dx_ocr'], [len(boxes1), len(boxes2)], 
                   color=['green', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('박스 개수')
    axes[0, 0].set_title('전체 박스 개수 비교')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, (label, count) in enumerate([('dxnn_detect', len(boxes1)), ('dx_ocr', len(boxes2))]):
        axes[0, 0].text(i, count + 10, str(count), ha='center', fontweight='bold')
    
    # 2. 박스 크기 히스토그램
    bins = np.logspace(0, 5, 50)
    axes[0, 1].hist(sizes1, bins=bins, alpha=0.5, label='dxnn_detect', color='green')
    axes[0, 1].hist(sizes2, bins=bins, alpha=0.5, label='dx_ocr', color='red')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('박스 면적 (pixels², log scale)')
    axes[0, 1].set_ylabel('개수')
    axes[0, 1].set_title('박스 크기 분포')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 작은 박스 누적 분포
    thresholds_fine = np.logspace(0, 4, 100)
    counts1 = [((sizes1 < t).sum()) for t in thresholds_fine]
    counts2 = [((sizes2 < t).sum()) for t in thresholds_fine]
    
    axes[1, 0].plot(thresholds_fine, counts1, label='dxnn_detect', color='green', linewidth=2)
    axes[1, 0].plot(thresholds_fine, counts2, label='dx_ocr', color='red', linewidth=2)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('박스 면적 threshold (pixels², log scale)')
    axes[1, 0].set_ylabel('작은 박스 개수 (< threshold)')
    axes[1, 0].set_title('작은 박스 누적 분포')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 차이 분석 (각 threshold별)
    diffs = np.array(counts2) - np.array(counts1)
    axes[1, 1].plot(thresholds_fine, diffs, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('박스 면적 threshold (pixels², log scale)')
    axes[1, 1].set_ylabel('박스 개수 차이 (dx_ocr - dxnn_detect)')
    axes[1, 1].set_title('차이 분석 (양수 = dx_ocr가 더 많음)')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].fill_between(thresholds_fine, 0, diffs, where=(diffs > 0), 
                            alpha=0.3, color='red', label='dx_ocr 더 많음')
    
    plt.tight_layout()
    output_path = output_dir / "size_analysis.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    logger.info(f"\n✓ 크기 분석 그래프 저장: {output_path}")
    
    # 결론
    logger.info("\n" + "=" * 80)
    logger.info("결론:")
    logger.info("=" * 80)
    
    small_box_diff = (sizes2 < 100).sum() - (sizes1 < 100).sum()
    small_box_pct = small_box_diff / (len(boxes2) - len(boxes1)) * 100 if len(boxes2) != len(boxes1) else 0
    
    logger.info(f"1. dx_ocr가 {len(boxes2) - len(boxes1)}개 더 많은 박스를 검출")
    logger.info(f"2. 이 중 면적 < 100px² 작은 박스: {small_box_diff}개 ({small_box_pct:.1f}%)")
    logger.info(f"3. 평균 박스 크기: dxnn_detect={sizes1.mean():.1f}, dx_ocr={sizes2.mean():.1f}")
    
    if sizes2.mean() < sizes1.mean():
        logger.warning("   → dx_ocr가 더 작은 박스들을 많이 검출 (노이즈 가능성)")
    
    # 주요 원인 추정
    logger.info("\n주요 원인 추정:")
    logger.info("1. ✗ dx_ocr는 min_side < 3 체크가 없어서 작은 노이즈 박스도 통과")
    logger.info("2. ✗ dx_ocr는 점수 계산이 달라서 threshold 통과 기준이 다를 수 있음")
    logger.info("3. ✗ dx_ocr는 unclip 후 재검증이 없어서 부정확한 박스도 포함")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_difference()
