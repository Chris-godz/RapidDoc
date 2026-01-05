# -*- encoding: utf-8 -*-
import math
import os
import platform
import traceback
import threading
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional, Callable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .logger import get_logger
from dx_engine import InferenceEngine

class DxInferSession:
    def __init__(self, config: Dict[str, Any], use_async: bool = False):
        """
        config에서 받는 항목:
        - model_path: 모델 파일 경로
        - engine_type: "dxengine"
        - use_cuda: GPU 사용 여부 (선택)
        - use_async: Async 모드 사용 여부
        """
        self.logger = get_logger("OrtInferSession")
        self.use_async = use_async

        model_path = config.get("model_path", None)
        self._verify_model(model_path)
        
        self.session = InferenceEngine(model_path)
        
        # Async callback 지원을 위한 추가 속성
        self.pending_requests = {}  # request_id -> (input_array, callback)
        self.lock = threading.Lock()
        self._request_counter = 0  # thread-safe 카운터
        
        # Async 모드일 때만 Callback 등록
        if self.use_async:
            self.session.register_callback(self._on_inference_complete)
            self.logger.info("DxInferSession initialized in ASYNC mode")
        else:
            self.logger.info("DxInferSession initialized in SYNC mode")

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    def _on_inference_complete(self, outputs: List[np.ndarray], user_arg: Any) -> int:
        """
        DX Engine 추론 완료 시 호출되는 콜백
        
        Args:
            outputs: DX Engine 출력 결과
            user_arg: (unique_id, callback)
        
        Returns:
            0 (success)
        """
        try:
            unique_id, callback = user_arg
            
            # 출력 결과는 numpy array
            result = outputs[0] if len(outputs) == 1 else outputs
            
            # 사용자 콜백 호출 (있는 경우)
            if callback is not None:
                callback(result, unique_id)
                
        except Exception as e:
            self.logger.error(f"Table inference callback error: {e}")
            import traceback
            traceback.print_exc()
        
    
    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """
        동기 추론 (run 사용)
        입력: numpy array (전처리된 이미지)
        출력: numpy array (모델 출력)
        
        table_structure_unet.py의 infer() 메서드에서 호출:
        result = self.session(input["img"][None, ...])[0][0]
        """
        return self.session.run(input_array) 
    
    def run_async(self, input_array: np.ndarray, callback: Optional[Callable] = None) -> int:
        """
        비동기 방식 추론 (callback 사용)
        run_async() 사용
        
        Args:
            input_array: 입력 데이터 (numpy array)
            callback: 완료 시 호출될 콜백 함수 (outputs, unique_id)
            
        Returns:
            request_id: DX Engine 요청 ID
        """
        # 고유 ID 생성 (thread-safe)
        with self.lock:
            unique_id = self._request_counter
            self._request_counter += 1
        
        # run_async 호출 (user_arg로 후처리 정보 전달)
        request_id = self.session.run_async(
            input_array if isinstance(input_array, list) else [input_array],
            user_arg=(unique_id, callback)
        )
        
        # pending requests에 저장
        with self.lock:
            self.pending_requests[request_id] = (input_array, callback)
        
        return request_id
    
    def wait_request(self, request_id: int) -> Optional[np.ndarray]:
        """
        특정 request의 완료를 대기
        
        Args:
            request_id: 대기할 요청 ID
            
        Returns:
            추론 결과 (callback이 설정되지 않은 경우에만)
        """
        outputs = self.session.wait(request_id)
        
        with self.lock:
            pending_data = self.pending_requests.pop(request_id, None)
        
        if pending_data is None:
            return None
            
        _, callback = pending_data
        
        # callback이 없는 경우 직접 결과 반환
        if callback is None:
            return outputs[0] if len(outputs) == 1 else outputs
        
        return None

