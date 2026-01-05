from pathlib import Path
from typing import Any, List, Optional, Tuple, Callable
import numpy as np
import threading
import queue

from ..utils.typings import RapidLayoutInput
from .base import InferSession
from ..utils.logger import Logger
from dx_engine import InferenceEngine
from onnxruntime import InferenceSession


class DXInferSession(InferSession):
    def __init__(self, cfg: RapidLayoutInput):
        super().__init__(cfg)
        self.logger = Logger(logger_name=__name__).get_log()
        # cfg에서 use_async 추출
        self.use_async = cfg.use_async if hasattr(cfg, 'use_async') else False

        if cfg.model_dir_or_path is None:
            raise ValueError("model_dir_or_path must be provided for DXEngine.")
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        self.model_path = model_path
        self.logger.info(f"Using {model_path}")

        self.session = InferenceEngine(str(self.model_path))
        self.sub_session = InferenceSession(str(cfg.sub_model_path))
        
        # Async callback 지원을 위한 추가 속성
        self.pending_requests = {}  # request_id -> (input_content, scale_factor, callback)
        self.lock = threading.Lock()
        self._request_counter = 0  # thread-safe 카운터
        
        # Async 모드일 때만 Callback 등록
        if self.use_async:
            self.session.register_callback(self._on_inference_complete)
            self.logger.info("DXInferSession initialized in ASYNC mode")
        else:
            self.logger.info("DXInferSession initialized in SYNC mode")
        
    def _on_inference_complete(self, outputs: List[np.ndarray], user_arg: Any) -> int:
        """
        DX Engine 추론 완료 시 호출되는 콜백
        
        Args:
            outputs: DX Engine 출력 결과
            user_arg: (unique_id, input_content, scale_factor, callback)
        
        Returns:
            0 (success)
        """
        try:
            unique_id, input_content, scale_factor, callback = user_arg
            
            # ONNX 후처리 실행
            shape = np.array(input_content.shape[1:-1])[None, ...].astype(np.float32)  # N, H, W, C -> H, W
            ort_feed = {
                "p2o.pd_op.concat.12.0": outputs[0],
                "p2o.pd_op.layer_norm.20.0": outputs[1],
                "im_shape": shape,
                "scale_factor": scale_factor
            }
            ort_outputs = self.sub_session.run(None, ort_feed)
            
            # 사용자 콜백 호출 (있는 경우)
            if callback is not None:
                callback(ort_outputs, unique_id)
                
        except Exception as e:
            self.logger.error(f"Layout inference callback error: {e}")
            import traceback
            traceback.print_exc()
    
    def __call__(self, input_content: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
        if self.use_async:
            self.request_id = self.run_async(input_content=input_content, scale_factor=scale_factor)
            return np.array([0])
        else:
            """
            동기 방식 추론 (기존 호환성 유지)
            일반 run() 사용
            """
            # Sync mode: 동기 run 호출
            outputs = self.session.run([input_content])
            
            shape = np.array(input_content.shape[1:-1])[None, ...].astype(np.float32)  # N, H, W, C -> H, W
            ort_feed = {
                "p2o.pd_op.concat.12.0": outputs[0],
                "p2o.pd_op.layer_norm.20.0": outputs[1],
                "im_shape": shape,
                "scale_factor": scale_factor
            }
            ort_outputs = self.sub_session.run(None, ort_feed)
            return ort_outputs
    
    def run_async(self, input_content: np.ndarray, scale_factor: np.ndarray, 
                  callback: Optional[Callable] = None) -> int:
        """
        비동기 방식 추론 (callback 사용)
        run_async() 사용
        
        Args:
            input_content: 입력 데이터
            scale_factor: 스케일 팩터
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
            [input_content], 
            user_arg=(unique_id, input_content, scale_factor, callback)
        )
        
        # pending requests에 저장
        with self.lock:
            self.pending_requests[request_id] = (input_content, scale_factor, callback)
        
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
            
        input_content, scale_factor, callback = pending_data
        
        # callback이 없는 경우 직접 후처리
        if callback is None:
            shape = np.array(input_content.shape[1:-1])[None, ...].astype(np.float32)
            ort_feed = {
                "p2o.pd_op.concat.12.0": outputs[0],
                "p2o.pd_op.layer_norm.20.0": outputs[1],
                "im_shape": shape,
                "scale_factor": scale_factor
            }
            return self.sub_session.run(None, ort_feed)
        
        return None

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character") -> List[str]:
        return []
