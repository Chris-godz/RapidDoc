#!/usr/bin/env python3
"""
JSON 파일에서 OCR Recognition 모델 입력 이미지 추출

이 스크립트는 demo1_model.json 파일을 읽어서:
1. 텍스트 영역(category_id=15)의 bbox 정보 추출
2. 원본 PDF에서 해당 영역을 크롭
3. OCR Recognition 모델 입력으로 사용할 수 있는 이미지 생성

사용법:
    python tools/extract_rec_inputs_from_json.py \
        --json_path demo/output-offline/demo1/auto/demo1_model.json \
        --pdf_path demo/output-offline/demo1/auto/demo1_origin.pdf \
        --output_dir demo/output-offline/demo1/rec_inputs \
        --max_samples 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from loguru import logger
import pypdfium2 as pdfium
from tqdm import tqdm


class RecInputExtractor:
    """JSON에서 OCR Recognition 입력 이미지 추출기"""
    
    def __init__(
        self,
        json_path: str,
        pdf_path: str,
        output_dir: str,
        category_ids: List[int] = [15],  # 텍스트 영역 category_id
        dpi: int = 200,
        padding: int = 2,
    ):
        """
        Args:
            json_path: demo1_model.json 경로
            pdf_path: 원본 PDF 경로
            output_dir: 출력 디렉토리
            category_ids: 추출할 category_id 리스트 (기본: [15] - 텍스트)
            dpi: PDF 렌더링 DPI
            padding: bbox 주변 패딩 (픽셀)
        """
        self.json_path = Path(json_path)
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.category_ids = category_ids
        self.dpi = dpi
        self.padding = padding
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 로드
        self.data = self._load_json()
        
        # PDF 로드
        self.pdf = pdfium.PdfDocument(str(self.pdf_path))
        logger.info(f"✓ PDF 로드 완료: {len(self.pdf)} 페이지")
        
    def _load_json(self) -> List[Dict]:
        """JSON 파일 로드"""
        logger.info(f"JSON 로딩: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✓ JSON 로드 완료: {len(data)} 페이지")
        return data
    
    def _parse_poly(self, poly: List[float]) -> Tuple[int, int, int, int]:
        """
        poly 좌표를 bbox로 변환
        
        Args:
            poly: [x1, y1, x2, y2, x3, y3, x4, y4] 형식
            
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        xs = poly[0::2]
        ys = poly[1::2]
        
        x_min = int(min(xs))
        y_min = int(min(ys))
        x_max = int(max(xs))
        y_max = int(max(ys))
        
        return x_min, y_min, x_max, y_max
    
    def _render_page(self, page_no: int) -> np.ndarray:
        """
        PDF 페이지를 이미지로 렌더링
        
        Args:
            page_no: 페이지 번호 (0-based)
            
        Returns:
            렌더링된 이미지 (H, W, 3)
        """
        page = self.pdf[page_no]
        
        # 렌더링
        pil_image = page.render(
            scale=self.dpi / 72,  # 72 DPI가 기본
            rotation=0,
        ).to_pil()
        
        # PIL to numpy
        img = np.array(pil_image)
        
        # RGB to BGR (OpenCV 형식)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def _crop_text_region(
        self,
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        page_info: Dict
    ) -> Optional[np.ndarray]:
        """
        이미지에서 텍스트 영역 크롭
        
        Args:
            img: 전체 페이지 이미지
            bbox: (x_min, y_min, x_max, y_max)
            page_info: 페이지 정보 (width, height)
            
        Returns:
            크롭된 이미지
        """
        x_min, y_min, x_max, y_max = bbox
        
        # JSON의 좌표는 원본 페이지 크기 기준이므로 스케일 조정
        json_width = page_info['width']
        json_height = page_info['height']
        img_height, img_width = img.shape[:2]
        
        scale_x = img_width / json_width
        scale_y = img_height / json_height
        
        # 스케일 조정
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        
        # 패딩 추가
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_max = min(img_width, x_max + self.padding)
        y_max = min(img_height, y_max + self.padding)
        
        # 크롭
        cropped = img[y_min:y_max, x_min:x_max]
        
        # 최소 크기 체크
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return None
        
        return cropped
    
    def extract_all(self, max_samples: Optional[int] = None) -> Dict:
        """
        모든 텍스트 영역 추출
        
        Args:
            max_samples: 최대 샘플 수 (None이면 전체)
            
        Returns:
            추출 결과 통계
        """
        logger.info("=" * 80)
        logger.info("OCR Recognition 입력 이미지 추출 시작")
        logger.info("=" * 80)
        
        stats = {
            'total_pages': len(self.data),
            'total_text_regions': 0,
            'extracted_images': 0,
            'skipped_images': 0,
            'by_page': []
        }
        
        sample_count = 0
        
        # 페이지별 처리
        for page_data in tqdm(self.data, desc="페이지 처리"):
            page_no = page_data['page_info']['page_no']
            page_info = page_data['page_info']
            
            logger.info(f"\n페이지 {page_no} 처리 중...")
            
            # 페이지 렌더링
            page_img = self._render_page(page_no)
            logger.debug(f"  렌더링 완료: {page_img.shape}")
            
            page_stats = {
                'page_no': page_no,
                'total_regions': 0,
                'extracted': 0,
                'skipped': 0
            }
            
            # 텍스트 영역 추출
            for idx, region in enumerate(page_data['layout_dets']):
                category_id = region['category_id']
                
                # 지정된 category_id만 처리
                if category_id not in self.category_ids:
                    continue
                
                stats['total_text_regions'] += 1
                page_stats['total_regions'] += 1
                
                # bbox 파싱
                poly = region['poly']
                bbox = self._parse_poly(poly)
                
                # 크롭
                cropped = self._crop_text_region(page_img, bbox, page_info)
                
                if cropped is None:
                    stats['skipped_images'] += 1
                    page_stats['skipped'] += 1
                    continue
                
                # 저장
                output_name = f"page{page_no:03d}_region{idx:04d}_cat{category_id}.jpg"
                output_path = self.output_dir / output_name
                
                cv2.imwrite(str(output_path), cropped)
                
                stats['extracted_images'] += 1
                page_stats['extracted'] += 1
                sample_count += 1
                
                # 최대 샘플 수 체크
                if max_samples and sample_count >= max_samples:
                    logger.info(f"\n최대 샘플 수({max_samples})에 도달했습니다.")
                    stats['by_page'].append(page_stats)
                    return stats
            
            stats['by_page'].append(page_stats)
            logger.info(f"  완료: {page_stats['extracted']}개 추출, {page_stats['skipped']}개 건너뜀")
        
        return stats
    
    def print_stats(self, stats: Dict):
        """통계 출력"""
        logger.info("\n" + "=" * 80)
        logger.info("추출 결과 통계")
        logger.info("=" * 80)
        logger.info(f"총 페이지 수: {stats['total_pages']}")
        logger.info(f"총 텍스트 영역 수: {stats['total_text_regions']}")
        logger.info(f"추출된 이미지 수: {stats['extracted_images']}")
        logger.info(f"건너뛴 이미지 수: {stats['skipped_images']}")
        
        logger.info("\n페이지별 상세:")
        for page_stat in stats['by_page']:
            logger.info(
                f"  페이지 {page_stat['page_no']:3d}: "
                f"{page_stat['extracted']:4d}개 추출, "
                f"{page_stat['skipped']:4d}개 건너뜀"
            )
        
        logger.info(f"\n출력 디렉토리: {self.output_dir}")
        logger.info("=" * 80)
    
    def create_metadata(self, stats: Dict):
        """메타데이터 파일 생성"""
        metadata = {
            'source_json': str(self.json_path),
            'source_pdf': str(self.pdf_path),
            'output_dir': str(self.output_dir),
            'category_ids': self.category_ids,
            'dpi': self.dpi,
            'padding': self.padding,
            'stats': stats
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 메타데이터 저장: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='JSON 파일에서 OCR Recognition 입력 이미지 추출'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='demo1_model.json 경로'
    )
    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='원본 PDF 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='rec_inputs',
        help='출력 디렉토리 (기본값: rec_inputs)'
    )
    parser.add_argument(
        '--category_ids',
        type=int,
        nargs='+',
        default=[15],
        help='추출할 category_id 리스트 (기본값: 15 - 텍스트)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='PDF 렌더링 DPI (기본값: 200)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=2,
        help='bbox 주변 패딩 픽셀 (기본값: 2)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='최대 추출 샘플 수 (기본값: 전체)'
    )
    
    args = parser.parse_args()
    
    # 경로 확인
    json_path = Path(args.json_path)
    pdf_path = Path(args.pdf_path)
    
    if not json_path.exists():
        logger.error(f"JSON 파일을 찾을 수 없습니다: {json_path}")
        return
    
    if not pdf_path.exists():
        logger.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    # 추출기 생성
    extractor = RecInputExtractor(
        json_path=str(json_path),
        pdf_path=str(pdf_path),
        output_dir=args.output_dir,
        category_ids=args.category_ids,
        dpi=args.dpi,
        padding=args.padding
    )
    
    # 추출 실행
    stats = extractor.extract_all(max_samples=args.max_samples)
    
    # 통계 출력
    extractor.print_stats(stats)
    
    # 메타데이터 저장
    extractor.create_metadata(stats)
    
    logger.info("\n✅ 완료!")


if __name__ == "__main__":
    main()
