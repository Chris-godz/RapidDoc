"""
pytest conftest — 무거운 의존성을 테스트 환경에서 격리하기 위한 sys.modules stub 설정.

rapid_doc.backend.pipeline.async_pipeline 을 import하면
model_init → rapid_layout_self → omegaconf 등 다수의 선택적 패키지가
필요하다. 이 파일이 없는 CI/개발 환경에서도 L1·L2 테스트가 통과할 수 있도록
pytest가 테스트 파일을 collect하기 전에 stub을 sys.modules에 주입한다.

실제 모델 실행(L3 통합 테스트)은 이 stub 없이 환경이 완전히 구성된 상태에서
실행한다.
"""

import sys
import types
from unittest.mock import MagicMock


def _make_stub(name: str) -> MagicMock:
    """이름을 가진 빈 패키지 stub을 생성하고 sys.modules에 등록한다.

    주의: spec=을 사용하지 않는다. spec을 지정하면 types.ModuleType에 없는
    속성(calculate_iou, ImageType 등)에 대해 AttributeError를 발생시켜
    'from stub_mod import name' 구문이 ImportError로 변환된다.
    """
    m = MagicMock()   # spec 없음 — 어떤 속성이든 MagicMock 반환
    m.__name__ = name
    m.__spec__ = None
    m.__path__ = []   # 패키지로 인식
    m.__file__ = f"<stub:{name}>"
    return m


def _inject(name: str) -> None:
    """name과 모든 부모 패키지 stub을 sys.modules에 주입한다."""
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        pkg = '.'.join(parts[:i])
        if pkg not in sys.modules:
            sys.modules[pkg] = _make_stub(pkg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 외부 패키지 stub (프로젝트 의존성이지만 개발 환경에 없을 수 있음)
# ─────────────────────────────────────────────────────────────────────────────
_EXTERNAL_STUBS = [
    'omegaconf',
    'colorlog',
    'shapely', 'shapely.geometry',
    'fast_langdetect',
    'rapidocr',
    'rapid_table',
    'onnxruntime', 'onnxruntime.capi', 'onnxruntime.capi.onnxruntime_pybind11_state',
    'dx_engine',
    'img2table', 'img2table.document',
    'img2table.tables', 'img2table.tables.processing',
    'torch',
]

for _name in _EXTERNAL_STUBS:
    _inject(_name)

# omegaconf는 DictConfig/OmegaConf 클래스가 직접 사용되므로 실체 제공
_oc = sys.modules['omegaconf']
_oc.DictConfig = dict
_oc.OmegaConf = MagicMock()

# ─────────────────────────────────────────────────────────────────────────────
# 2. 내부 heavy 모듈 전체 교체 — 실제 파일은 모델 파일 / 하드웨어 드라이버에 의존
# ─────────────────────────────────────────────────────────────────────────────
# model_init: MineruPipelineModel, AtomModelSingleton, custom_model_init 제공
_model_init = _make_stub('rapid_doc.backend.pipeline.model_init')

class _MineruPipelineModel:  # noqa: D101 — stub
    def __init__(self, **kwargs):  # pipeline_analyze calls MineruPipelineModel(**model_input)
        pass
    layout_model = MagicMock()
    formula_model = MagicMock()
    ocr_model = MagicMock()
    table_model = MagicMock()

class _AtomModelSingleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def get_atom_model(self, atom_model_name=None, **kwargs):
        return MagicMock()

def _custom_model_init(**kwargs):
    return _MineruPipelineModel()

_model_init.MineruPipelineModel = _MineruPipelineModel
_model_init.AtomModelSingleton = _AtomModelSingleton
_model_init.custom_model_init = _custom_model_init
sys.modules['rapid_doc.backend.pipeline.model_init'] = _model_init

# model_list: AtomicModel enum
_model_list = _make_stub('rapid_doc.backend.pipeline.model_list')
from enum import Enum as _Enum
class _AtomicModel(_Enum):
    OCR = 'ocr'
    LAYOUT = 'layout'
    FORMULA = 'formula'
    TABLE = 'table'
_model_list.AtomicModel = _AtomicModel
sys.modules['rapid_doc.backend.pipeline.model_list'] = _model_list

# ─────────────────────────────────────────────────────────────────────────────
# 3. 내부 util stub — L1/L2 용 (pypdfium2 없는 환경 대비)
# ─────────────────────────────────────────────────────────────────────────────
# span_pre_proc → pdf_image_tools → pypdfium2 의존 체인이 있으므로
# pypdfium2 가 없는 환경(시스템 Python, CI)에서도 async_pipeline 이 임포트되도록
# 내부 util 도 stub 처리한다.
#
# L3 통합 테스트(TestIntegrationSyncVsAsync.setUpClass)에서는 이 stub 들을
# sys.modules 에서 제거한 뒤 실제 모듈을 재임포트한다.

# config_reader
_config_reader = _make_stub('rapid_doc.utils.config_reader')
_config_reader.get_formula_enable = lambda x: bool(x)
_config_reader.get_table_enable = lambda x: bool(x)
_config_reader.get_device = lambda: 'cpu'
sys.modules['rapid_doc.utils.config_reader'] = _config_reader

# enum_class: 순수 클래스 정의 — 실제 모듈 사용 (ImageType, MakeMode 등 포함)
# (stub 처리 안 함)

# model_utils
_model_utils = _make_stub('rapid_doc.utils.model_utils')
def _crop_img(res, np_img, crop_paste_x=0, crop_paste_y=0):
    import numpy as _np
    h, w = (np_img.shape[:2] if np_img is not None else (64, 64))
    return _np.zeros((max(h // 4, 1), max(w // 4, 1), 3), dtype=_np.uint8), []
_model_utils.crop_img = _crop_img
_model_utils.get_res_list_from_layout_res = MagicMock(return_value=([], [], []))
_model_utils.clean_memory = MagicMock()
_model_utils.get_vram = MagicMock(return_value=0)
sys.modules['rapid_doc.utils.model_utils'] = _model_utils

# ocr_utils
_ocr_utils = _make_stub('rapid_doc.utils.ocr_utils')
_ocr_utils.merge_det_boxes = MagicMock(side_effect=lambda x: x)
_ocr_utils.update_det_boxes = MagicMock(side_effect=lambda boxes, adj: boxes)
_ocr_utils.sorted_boxes = MagicMock(side_effect=lambda x: x)
_ocr_utils.get_adjusted_mfdetrec_res = MagicMock(return_value=[])
_ocr_utils.get_ocr_result_list = MagicMock(return_value=[])
_ocr_utils.OcrConfidence = MagicMock()
_ocr_utils.get_ocr_result_list_table = MagicMock(return_value=[])
sys.modules['rapid_doc.utils.ocr_utils'] = _ocr_utils

# span_pre_proc (pdf_image_tools → pypdfium2 체인을 차단)
_span = _make_stub('rapid_doc.utils.span_pre_proc')
_span.txt_spans_bbox_extract = MagicMock(return_value=None)
_span.extract_table_fill_image = MagicMock(return_value=None)
_span.txt_most_angle_extract_table = MagicMock(return_value=0)
sys.modules['rapid_doc.utils.span_pre_proc'] = _span

# boxbase
_boxbase = _make_stub('rapid_doc.utils.boxbase')
_boxbase.rotate_image_and_boxes = MagicMock(side_effect=lambda img, boxes, angle: (img, boxes))
sys.modules['rapid_doc.utils.boxbase'] = _boxbase

# checkbox_det_cls
_checkbox = _make_stub('rapid_doc.utils.checkbox_det_cls')
_checkbox.checkbox_predict = MagicMock(return_value=[])
sys.modules['rapid_doc.utils.checkbox_det_cls'] = _checkbox

# ─────────────────────────────────────────────────────────────────────────────
# 공개 API: L3 통합 테스트에서 제거할 stub 키 목록
# ─────────────────────────────────────────────────────────────────────────────

# L1/L2에서 주입한 외부 패키지 stub 전체 목록.
# setUpClass에서 이 키들을 sys.modules에서 제거하면 실제 설치된 패키지가 사용된다.
EXTERNAL_STUBS: list[str] = [
    'omegaconf',
    'colorlog',
    'shapely', 'shapely.geometry',
    'fast_langdetect',
    'rapidocr',
    'rapid_table',
    'onnxruntime', 'onnxruntime.capi', 'onnxruntime.capi.onnxruntime_pybind11_state',
    'dx_engine',
    'img2table', 'img2table.document',
    'img2table.tables', 'img2table.tables.processing',
    'torch',
]

INTERNAL_UTIL_STUBS: list[str] = [
    'rapid_doc.utils.config_reader',
    'rapid_doc.utils.model_utils',
    'rapid_doc.utils.ocr_utils',
    'rapid_doc.utils.span_pre_proc',
    'rapid_doc.utils.boxbase',
    'rapid_doc.utils.checkbox_det_cls',
    'rapid_doc.backend.pipeline.model_init',
    'rapid_doc.backend.pipeline.model_list',
    'rapid_doc.backend.pipeline.async_pipeline',
    'rapid_doc.backend.pipeline.batch_analyze',
    'rapid_doc.backend.pipeline.pipeline_analyze',
]

# L3 통합 테스트에서 sys.modules를 완전히 초기화할 때 사용하는 전체 목록
ALL_CONFTEST_STUBS: list[str] = EXTERNAL_STUBS + INTERNAL_UTIL_STUBS
