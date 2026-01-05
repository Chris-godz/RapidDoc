"""
비동기 파이프라인 처리 모듈
DX Engine의 run_async() 기능을 활용한 비동기 문서 분석
"""
import time
import threading
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from PIL import Image
from loguru import logger

from .model_init import MineruPipelineModel
from .batch_analyze import BatchAnalyze
from ...utils.config_reader import get_device


class AsyncBatchAnalyzer:
    """비동기 배치 분석기"""
    
    def __init__(
        self,
        model: MineruPipelineModel,
        batch_ratio: int,
        formula_enable: bool,
        table_enable: bool,
        layout_config: dict = None,
        ocr_config: dict = None,
        formula_config: dict = None,
        table_config: dict = None,
        checkbox_config: dict = None,
        input_interval: float = 0.0,
        verbose: bool = False
    ):
        """
        Args:
            model: 초기화된 MineruPipelineModel
            batch_ratio: 배치 비율
            formula_enable: 수식 인식 활성화 여부
            table_enable: 테이블 인식 활성화 여부
            layout_config: 레이아웃 설정
            ocr_config: OCR 설정
            formula_config: 수식 설정
            table_config: 테이블 설정
            checkbox_config: 체크박스 설정
            input_interval: 비동기 작업 제출 간 대기 시간 (초)
            verbose: 상세 로그 출력 여부
        """
        self.model = model
        self.batch_ratio = batch_ratio
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.layout_config = layout_config
        self.ocr_config = ocr_config
        self.formula_config = formula_config
        self.table_config = table_config
        self.checkbox_config = checkbox_config
        self.input_interval = input_interval
        self.verbose = verbose
        
        # 결과 저장소
        self.results = {}  # page_idx -> result
        self.pending_count = 0
        self.completed_count = 0
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        
        # 성능 통계
        self.pdf_perf_stats = defaultdict(lambda: defaultdict(lambda: {'time': 0.0, 'count': 0}))
        
        # BatchAnalyze 인스턴스 생성 (sync 모드)
        self.batch_analyzer = BatchAnalyze(
            model=model,
            batch_ratio=batch_ratio,
            formula_enable=formula_enable,
            table_enable=table_enable,
            enable_ocr_det_batch=True,
            layout_config=layout_config,
            ocr_config=ocr_config,
            formula_config=formula_config,
            table_config=table_config,
            checkbox_config=checkbox_config
        )
        
    def process_pages(
        self, 
        images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict, int, int]]
    ) -> Tuple[List[Any], Dict[int, Dict[str, Dict[str, float]]]]:
        """
        페이지들을 비동기로 처리
        
        Args:
            images_with_extra_info: (img, scale, ocr_enable, lang, page_dict, pdf_idx, page_idx) 리스트
            
        Returns:
            (결과 리스트, PDF별 성능 통계)
        """
        total_pages = len(images_with_extra_info)
        logger.info(f"🚀 AsyncBatchAnalyzer: Processing {total_pages} pages")
        
        start_time = time.time()
        
        # 현재는 sync 방식으로 처리 (BatchAnalyze 재사용)
        # TODO: DX Engine의 run_async를 활용한 진정한 비동기 처리 구현
        
        # 임시: 기존 batch_analyzer를 사용하여 동기 처리
        results, page_stats = self.batch_analyzer(images_with_extra_info)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ AsyncBatchAnalyzer: Completed {total_pages} pages in {elapsed:.2f}s")
        
        return results, page_stats


def async_batch_image_analyze(
    images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict, int, int]],
    formula_enable: bool = True,
    table_enable: bool = True,
    layout_config: dict = None,
    ocr_config: dict = None,
    formula_config: dict = None,
    table_config: dict = None,
    checkbox_config: dict = None,
    input_interval: float = 0.0,
    verbose: bool = False
) -> Tuple[List[Any], Dict[int, Dict[str, Dict[str, float]]]]:
    """
    비동기 배치 이미지 분석
    
    Args:
        images_with_extra_info: (img, scale, ocr_enable, lang, page_dict, pdf_idx, page_idx) 리스트
        formula_enable: 수식 인식 활성화
        table_enable: 테이블 인식 활성화
        layout_config: 레이아웃 설정
        ocr_config: OCR 설정
        formula_config: 수식 설정
        table_config: 테이블 설정
        checkbox_config: 체크박스 설정
        input_interval: 비동기 작업 제출 간 대기 시간
        verbose: 상세 로그 출력
        
    Returns:
        (결과 리스트, PDF별 성능 통계)
    """
    from .pipeline_analyze import custom_model_init
    from ...utils.model_utils import get_vram, clean_memory
    import os
    
    # 모델 초기화 (async 모드로 설정)
    # OCR config에 use_async 추가
    if ocr_config is None:
        ocr_config = {}
    ocr_config['use_async'] = True
    
    if layout_config is None:
        layout_config = {}
    layout_config['use_async'] = True
    
    if table_config is None:
        table_config = {}
    table_config['use_async'] = True
    
    model = custom_model_init(
        lang=None,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
    )
    
    # batch_ratio 계산
    batch_ratio = 1
    device = get_device()
    
    if str(device).startswith('npu') or str(device).startswith('cuda'):
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('MINERU_VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
        else:
            batch_ratio = 1
            logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_ratio}')
    
    # AsyncBatchAnalyzer 생성
    analyzer = AsyncBatchAnalyzer(
        model=model,
        batch_ratio=batch_ratio,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config=checkbox_config,
        input_interval=input_interval,
        verbose=verbose
    )
    
    # 페이지 처리
    results, page_stats = analyzer.process_pages(images_with_extra_info)
    
    # 메모리 정리
    clean_memory(device)
    
    return results, page_stats
