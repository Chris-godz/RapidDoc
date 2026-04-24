import html
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import inline_left_delimiter, inline_right_delimiter
from rapid_doc.model.table.rapid_table_self.table_cls import TableCls
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput
from rapid_doc.model.layout.rapid_layout_self import RapidLayoutInput, RapidLayout, ModelType as LayoutModelType
from rapid_doc.utils.boxbase import is_in
from rapid_doc.utils.config_reader import get_device
from rapid_doc.utils.ocr_utils import points_to_bbox, bbox_to_points


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class RapidTableModel(object):
    def __init__(self, ocr_engine, table_config=None, use_async=False):
        if table_config is None:
            table_config = {}
        self.use_async = use_async
        device = get_device()
        engine_cfg = None
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": device_id}
        self.model_type = table_config.get("model_type", ModelType.UNET_SLANET_PLUS)
        self.ocr_engine = ocr_engine

        if self.model_type == ModelType.SLANEXT:
            # 有线/无线 단元格识别
            self.table_cls = self._try_load_table_cls(table_config)
            wired_cell_args = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRED_TABLE_CELL_DET,
                                               model_dir_or_path=table_config.get("wired_cell.model_dir_or_path"),
                                               conf_thresh=0.3,
                                               engine_cfg=engine_cfg or {},
                                               use_async=self.use_async)
            self.wired_table_cell = RapidLayout(cfg=wired_cell_args)
            wireless_cell_args = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRELESS_TABLE_CELL_DET,
                                                  model_dir_or_path=table_config.get("wireless_cell.model_dir_or_path"),
                                                  conf_thresh=0.3,
                                                  engine_cfg=engine_cfg or {},
                                                  use_async=self.use_async)
            self.wireless_table_cell = RapidLayout(cfg=wireless_cell_args)
            # 有线/无线 表结构识别
            wired_input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRED, use_ocr=False,
                                               model_dir_or_path=table_config.get("wired_table.model_dir_or_path"),
                                               engine_cfg=engine_cfg or {},
                                               use_async=self.use_async)
            self.wired_table_model = RapidTable(wired_input_args)
            wireless_input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRELESS, use_ocr=False,
                                                  model_dir_or_path=table_config.get("wireless_table.model_dir_or_path"),
                                                  engine_cfg=engine_cfg or {},
                                                  use_async=self.use_async)
            self.wireless_table_model = RapidTable(wireless_input_args)
        elif self.model_type == ModelType.UNET_SLANET_PLUS:
            self.table_cls = self._try_load_table_cls(table_config)
            engine_type = table_config.get("engine_type")
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, use_ocr=False,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg or {},
                                               engine_type=engine_type,
                                               use_async=self.use_async)
            self.wired_table_model = RapidTable(wired_input_args)
            wireless_type_val = table_config.get("wireless_model_type", ModelType.SLANETPLUS.value)
            wireless_model_type = ModelType(wireless_type_val) if isinstance(wireless_type_val, str) else wireless_type_val
            wireless_engine_type = table_config.get("wireless_engine_type")
            wireless_input_args = RapidTableInput(model_type=wireless_model_type, use_ocr=False,
                                                  model_dir_or_path=table_config.get("slanet_plus.model_dir_or_path"),
                                                  engine_cfg=engine_cfg or {},
                                                  engine_type=wireless_engine_type,
                                                  use_async=self.use_async)
            self.wireless_table_model = RapidTable(wireless_input_args)
        elif self.model_type == ModelType.UNET_UNITABLE:
            self.table_cls = self._try_load_table_cls(table_config)
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, use_ocr=False,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg or {}, )
            self.wired_table_model = RapidTable(wired_input_args)
            wireless_input_args = RapidTableInput(model_type=ModelType.UNITABLE, use_ocr=False,
                                                  model_dir_or_path=table_config.get("unitable.model_dir_or_path"),
                                                  engine_cfg=engine_cfg or {}, )
            self.wireless_table_model = RapidTable(wireless_input_args)
        else:
            input_args = RapidTableInput(model_type=self.model_type, use_ocr=False,
                                         model_dir_or_path=table_config.get("model_dir_or_path"),
                                         engine_cfg=engine_cfg or {}, engine_type=table_config.get("engine_type"), )
            self.table_model = RapidTable(input_args)

    @staticmethod
    def _try_load_table_cls(table_config):
        """Try to load the table classification model (paddle_cls.onnx). Returns None on failure."""
        cls_model_path = table_config.get("cls.model_dir_or_path")
        if cls_model_path is not None and not Path(cls_model_path).exists():
            logger.warning(
                f"Table classification model file not found: {cls_model_path}. "
                "All tables will be treated as wired tables."
            )
            return None
        try:
            return TableCls(model_path=cls_model_path)
        except Exception as e:
            logger.warning(
                f"Failed to load table classification model: {e}. "
                "All tables will be treated as wired tables."
            )
            return None

    def _classify_table(self, image):
        """Classify the table type. Returns 'wired' if table_cls is unavailable."""
        if self.table_cls is None:
            return "wired", 0.0
        return self.table_cls(image)

    def predict(self, image, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, use_img2table=False):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:

            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            # Check if table is rotated by analyzing text box aspect ratios
            is_rotated = False
            if det_res:
                vertical_count = 0

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if vertical_count >= len(det_res) * 0.3:
                    is_rotated = True

                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

            # Rotate image if necessary
            if is_rotated:
                # logger.debug("Table appears to be in portrait orientation, rotating 90 degrees clockwise")
                image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_CLOCKWISE)
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Continue with OCR on potentially rotated image
        if not ocr_result:
            ocr_result = self.ocr_engine.ocr(bgr_image, mfd_res=mfd_res)[0]
            if ocr_result:
                ocr_result = [list(x) for x in zip(*[[item[0], item[1][0], item[1][1]] for item in ocr_result])]
            else:
                ocr_result = None
        if not ocr_result:
            return None, None, None, None
        # 把图片结果，添加到ocr_result里。uuid作为占位符，后面保存图片时替换
        if fill_image_res:
            for fill_image in fill_image_res:
                bbox = points_to_bbox(fill_image['ocr_bbox'])
                cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), thickness=-1) # 填白图像区域，防止表格识别被影响
                ocr_result[0].append(fill_image['ocr_bbox'])
                ocr_result[1].append(fill_image['uuid'])
                ocr_result[2].append(1)
                if skip_text_in_image:
                    # 找出所有 OCR 框在图片框内的下标
                    delete_indices = []
                    for idx, ocr in enumerate(ocr_result[0][:-1]):  # 排除刚添加的图片框自身
                        if is_in(points_to_bbox(ocr), points_to_bbox(fill_image['ocr_bbox'])):
                            delete_indices.append(idx)
                    # 按逆序删除，防止下标错位
                    for idx in sorted(delete_indices, reverse=True):
                        del ocr_result[0][idx]
                        del ocr_result[1][idx]
                        del ocr_result[2][idx]
        # 表格内的公式填充
        if mfd_res:
            for mfd in mfd_res:
                if mfd.get('latex'):
                    ocr_result[1].append(f"{inline_left_delimiter}{mfd['latex']}{inline_right_delimiter}")
                elif mfd.get('checkbox'):
                    ocr_result[1].append(mfd['checkbox'])
                else:
                    continue
                ocr_result[0].append(bbox_to_points(mfd['bbox']))
                ocr_result[2].append(1)

        """开始识别表格"""
        cls = None
        """使用 img2table 识别"""
        if use_img2table:
            try:
                from rapid_doc.model.table.img2table_self.image import Image
                from rapid_doc.model.table.img2table_self.RapidOcrTable import RapidOcrTable

                cls, elasp = self._classify_table(image)
                if cls == "wired":
                    borderless_tables = False
                else:
                    borderless_tables = True
                opencv_ocr = RapidOcrTable(ocr_result)
                doc = Image(src=bgr_image)
                extracted_tables = doc.extract_tables(
                    ocr=opencv_ocr,
                    implicit_rows=False,
                    implicit_columns=False,
                    borderless_tables=borderless_tables,
                    min_confidence=50
                )
                if extracted_tables:
                    # print(f"img2table detected {len(extracted_tables)} tables")
                    html_code = "<html><body>" + extracted_tables[0].html + "</body></html>"
                    return html_code, None, None, None
            except ImportError:
                raise ValueError(
                    "Could not import img2table python package. "
                    "Please install it with `pip install img2table`."
                )
            except Exception as e:
                logger.exception(e)

        """使用 rapid_table_self 识别"""
        try:
            _WIRELESS_PLACEHOLDER_HTML = (
                "<table><tr><td>wireless table recognition is not supported</td></tr></table>"
            )

            if self.model_type == ModelType.SLANEXT:
                if not cls:
                    cls, elasp = self._classify_table(bgr_image)
                if cls == "wired":
                    cell_res = self.wired_table_cell([bgr_image])
                    model_runner = (self.wired_table_model)
                    cell_results = (cell_res[0].boxes, cell_res[0].scores)
                    table_results = model_runner(bgr_image, ocr_result, cell_results=cell_results)
                else:  # wireless — skip inference
                    logger.info("Wireless table detected — skipping inference")
                    return _WIRELESS_PLACEHOLDER_HTML, [], [], 0
            elif self.model_type == ModelType.UNET_SLANET_PLUS or self.model_type == ModelType.UNET_UNITABLE:
                if not cls:
                    cls, elasp = self._classify_table(bgr_image)
                if cls == "wired":
                    table_results = self.wired_table_model(bgr_image, ocr_result)
                else:  # wireless — skip inference
                    logger.info("Wireless table detected — skipping inference")
                    return _WIRELESS_PLACEHOLDER_HTML, [], [], 0
            else:
                table_results = self.table_model(bgr_image, ocr_result)

            html_code = table_results.pred_html
            table_cell_bboxes = table_results.cell_bboxes
            logic_points = table_results.logic_points
            elapse = table_results.elapse
            return html_code, table_cell_bboxes, logic_points, elapse
        except Exception as e:
            logger.exception(e)
            return None, None, None, None

