"""
두 Detection Postprocessing 함수의 출력 비교 스크립트

dxnn_detect.py의 DXNNDetectionVisualizer와 
dx_ocr.py의 DxTextDetector의 출력을 비교합니다.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from value_compare.detect.dxnn_detect import DXNNDetectionVisualizer
from rapid_doc.model.ocr.dx_ocr import DxTextDetector


def compare_boxes(boxes1, boxes2, tolerance=1.0):
    """
    두 박스 세트를 비교
    
    Args:
        boxes1: 첫 번째 박스 배열 (N, 4, 2)
        boxes2: 두 번째 박스 배열 (M, 4, 2)
        tolerance: 좌표 차이 허용 오차 (픽셀)
        
    Returns:
        dict: 비교 결과
    """
    if boxes1 is None and boxes2 is None:
        return {
            "identical": True,
            "boxes1_count": 0,
            "boxes2_count": 0,
            "message": "Both returned None"
        }
    
    if boxes1 is None or boxes2 is None:
        return {
            "identical": False,
            "boxes1_count": 0 if boxes1 is None else len(boxes1),
            "boxes2_count": 0 if boxes2 is None else len(boxes2),
            "message": f"One is None: boxes1={'None' if boxes1 is None else len(boxes1)}, boxes2={'None' if boxes2 is None else len(boxes2)}"
        }
    
    # 박스 개수 비교
    if len(boxes1) != len(boxes2):
        return {
            "identical": False,
            "boxes1_count": len(boxes1),
            "boxes2_count": len(boxes2),
            "message": f"Different box counts: {len(boxes1)} vs {len(boxes2)}"
        }
    
    # 각 박스 비교 (순서가 같다고 가정)
    max_diff = 0.0
    total_diff = 0.0
    diff_count = 0
    
    for i, (box1, box2) in enumerate(zip(boxes1, boxes2)):
        diff = np.abs(box1 - box2)
        box_max_diff = diff.max()
        box_mean_diff = diff.mean()
        
        max_diff = max(max_diff, box_max_diff)
        total_diff += box_mean_diff
        diff_count += 1
        
        if box_max_diff > tolerance:
            logger.warning(f"Box {i}: max_diff={box_max_diff:.2f}, mean_diff={box_mean_diff:.2f}")
    
    avg_diff = total_diff / diff_count if diff_count > 0 else 0.0
    
    is_identical = max_diff <= tolerance
    
    return {
        "identical": is_identical,
        "boxes1_count": len(boxes1),
        "boxes2_count": len(boxes2),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "message": f"Max diff: {max_diff:.4f}, Avg diff: {avg_diff:.4f}" + (" ✓ PASS" if is_identical else " ✗ FAIL")
    }


def visualize_comparison(img, boxes1, boxes2, output_path):
    """
    두 결과를 시각적으로 비교
    
    Args:
        img: 원본 이미지
        boxes1: dxnn_detect 결과
        boxes2: dx_ocr 결과
        output_path: 출력 경로
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. dxnn_detect 결과
    img1 = img.copy()
    if boxes1 is not None:
        for i, box in enumerate(boxes1):
            box = box.astype(np.int32)
            cv2.polylines(img1, [box], True, (0, 255, 0), 2)
            center = box.mean(axis=0).astype(np.int32)
            cv2.putText(img1, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'dxnn_detect.py\n({len(boxes1) if boxes1 is not None else 0} boxes)', 
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. dx_ocr 결과
    img2 = img.copy()
    if boxes2 is not None:
        for i, box in enumerate(boxes2):
            box = box.astype(np.int32)
            cv2.polylines(img2, [box], True, (0, 0, 255), 2)
            center = box.mean(axis=0).astype(np.int32)
            cv2.putText(img2, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'dx_ocr.py\n({len(boxes2) if boxes2 is not None else 0} boxes)', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. 오버레이 (Green=dxnn_detect, Red=dx_ocr)
    img3 = img.copy()
    if boxes1 is not None:
        for box in boxes1:
            box = box.astype(np.int32)
            cv2.polylines(img3, [box], True, (0, 255, 0), 2)
    if boxes2 is not None:
        for box in boxes2:
            box = box.astype(np.int32)
            cv2.polylines(img3, [box], True, (0, 0, 255), 1)
    
    axes[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay\n(Green=dxnn_detect, Red=dx_ocr)', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ 비교 시각화 저장: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Detection Postprocessing Output Comparison")
    parser.add_argument("--model_path", type=str, required=True, help="DXNN 모델 경로")
    parser.add_argument("--image_path", type=str, required=True, help="테스트 이미지 경로")
    parser.add_argument("--output_dir", type=str, default="value_compare/detect/output_compare", help="출력 디렉토리")
    parser.add_argument("--box_thresh", type=float, default=0.3, help="박스 임계값")
    parser.add_argument("--unclip_ratio", type=float, default=1.8, help="박스 확장 비율")
    parser.add_argument("--use_dilation", action="store_true", help="dx_ocr에서 팽창 연산 사용")
    parser.add_argument("--tolerance", type=float, default=1.0, help="좌표 차이 허용 오차 (픽셀)")
    
    args = parser.parse_args()
    
    # 경로 설정
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 존재 확인
    if not model_path.exists():
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    if not image_path.exists():
        logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    logger.info("=" * 80)
    logger.info("Detection Postprocessing Output Comparison")
    logger.info("=" * 80)
    logger.info(f"모델: {model_path}")
    logger.info(f"이미지: {image_path}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"Box threshold: {args.box_thresh}")
    logger.info(f"Unclip ratio: {args.unclip_ratio}")
    logger.info(f"Use dilation (dx_ocr): {args.use_dilation}")
    logger.info(f"Tolerance: {args.tolerance} pixels")
    logger.info("=" * 80)
    
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    logger.info(f"이미지 크기: {img.shape[1]}x{img.shape[0]}")
    
    # 1. dxnn_detect.py (DXNNDetectionVisualizer)
    logger.info("\n[1/2] dxnn_detect.py 실행 중...")
    visualizer1 = DXNNDetectionVisualizer(
        model_path=str(model_path),
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio
    )
    
    boxes1, prob_map1, elapse1 = visualizer1.detect(img)
    logger.info(f"  - 검출 시간: {elapse1:.4f}초")
    logger.info(f"  - 검출 박스: {len(boxes1) if boxes1 is not None else 0}개")
    
    # 메모리 절약을 위해 첫 번째 모델 삭제
    del visualizer1.engine
    del visualizer1
    logger.info("  - dxnn_detect 모델 메모리 해제")
    
    # 2. dx_ocr.py (DxTextDetector)
    logger.info("\n[2/2] dx_ocr.py 실행 중...")
    detector2 = DxTextDetector(
        model_path=str(model_path),
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
        use_dilation=args.use_dilation
    )
    
    det_result = detector2(img)
    boxes2 = det_result.boxes
    elapse2 = det_result.elapse
    logger.info(f"  - 검출 시간: {elapse2:.4f}초")
    logger.info(f"  - 검출 박스: {len(boxes2) if boxes2 is not None else 0}개")
    
    # 3. 결과 비교
    logger.info("\n" + "=" * 80)
    logger.info("결과 비교")
    logger.info("=" * 80)
    
    comparison = compare_boxes(boxes1, boxes2, tolerance=args.tolerance)
    
    logger.info(f"박스 개수:")
    logger.info(f"  - dxnn_detect.py: {comparison['boxes1_count']}")
    logger.info(f"  - dx_ocr.py:      {comparison['boxes2_count']}")
    
    if comparison['identical']:
        logger.success(f"✓ 두 함수의 출력이 동일합니다! (tolerance={args.tolerance}px)")
    else:
        logger.warning(f"✗ 두 함수의 출력이 다릅니다!")
    
    logger.info(f"메시지: {comparison['message']}")
    
    if 'max_diff' in comparison:
        logger.info(f"최대 좌표 차이: {comparison['max_diff']:.4f} pixels")
        logger.info(f"평균 좌표 차이: {comparison['avg_diff']:.4f} pixels")
    
    # 4. 시각화
    logger.info("\n시각화 생성 중...")
    output_filename = image_path.stem
    comparison_path = output_dir / f"{output_filename}_comparison.png"
    visualize_comparison(img, boxes1, boxes2, comparison_path)
    
    # 5. 상세 비교 결과 저장
    if boxes1 is not None and boxes2 is not None and len(boxes1) == len(boxes2):
        logger.info("\n상세 박스 좌표 비교:")
        for i, (box1, box2) in enumerate(zip(boxes1, boxes2)):
            diff = np.abs(box1 - box2)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            if max_diff > args.tolerance:
                logger.warning(f"  Box {i}: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
                logger.debug(f"    dxnn_detect: {box1.flatten()}")
                logger.debug(f"    dx_ocr:      {box2.flatten()}")
    
    # 6. 개별 출력 저장 (디버깅용)
    logger.info("\n개별 출력 저장 중...")
    
    # dxnn_detect 결과 저장
    if boxes1 is not None:
        boxes_path = output_dir / f"{output_filename}_dxnn_detect_boxes.npy"
        np.save(str(boxes_path), boxes1)
        logger.info(f"✓ dxnn_detect boxes 저장: {boxes_path}")
    
    if prob_map1 is not None:
        prob_map_path = output_dir / f"{output_filename}_prob_map.npy"
        np.save(str(prob_map_path), prob_map1)
        logger.info(f"✓ probability map 저장: {prob_map_path}")
    
    # dx_ocr 결과 저장
    if boxes2 is not None:
        boxes_path = output_dir / f"{output_filename}_dx_ocr_boxes.npy"
        np.save(str(boxes_path), boxes2)
        logger.info(f"✓ dx_ocr boxes 저장: {boxes_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ 모든 작업 완료!")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info("=" * 80)
    
    # 반환값으로 비교 결과 제공
    return comparison


if __name__ == "__main__":
    result = main()
    
    # Exit code 설정
    if result is not None and result['identical']:
        sys.exit(0)  # 성공
    else:
        sys.exit(1)  # 실패
