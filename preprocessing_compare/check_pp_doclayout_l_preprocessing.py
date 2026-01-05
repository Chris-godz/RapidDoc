"""
PP-DocLayout-L 모델의 Preprocessing 출력 확인 스크립트

이 스크립트는 pp_doclayout_l 모델의 전처리 과정에서 생성되는
중간 출력물들을 시각화하고 저장합니다.

사용법:
    python preprocessing_compare/check_pp_doclayout_l_preprocessing.py --image <이미지 경로>
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pypdfium2 import PdfDocument

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType
from rapid_doc.utils.pdf_reader import page_to_image


class PreprocessingVisualizer:
    """PP-DocLayout-L 전처리 과정을 시각화하는 클래스"""
    
    def __init__(self, model_type: ModelType = ModelType.PP_DOCLAYOUT_L):
        """
        Args:
            model_type: 모델 타입 (기본값: PP_DOCLAYOUT_L)
        """
        self.model_type = model_type
        
        # 모델 타입에 따른 이미지 크기 설정
        if model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
            self.img_size = (800, 800)
        elif model_type == ModelType.PP_DOCLAYOUT_S:
            self.img_size = (480, 480)
        else:  # PP_DOCLAYOUT_L, PP_DOCLAYOUT_M
            self.img_size = (640, 640)
        
        self.preprocessor = PPPreProcess(img_size=self.img_size, model_type=model_type)
        
        print(f"모델 타입: {model_type.value}")
        print(f"입력 크기: {self.img_size}")
        print(f"Mean: {self.preprocessor.mean}")
        print(f"Std: {self.preprocessor.std}")
        print(f"Scale: {self.preprocessor.scale}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """이미지 파일을 로드합니다."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        # BGR -> RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def load_pdf_page(self, pdf_path: str, page_num: int = 0, dpi: int = 200) -> np.ndarray:
        """
        PDF 파일의 특정 페이지를 이미지로 로드합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            page_num: 페이지 번호 (0부터 시작)
            dpi: 렌더링 DPI (기본값: 200)
            
        Returns:
            np.ndarray: RGB 이미지
        """
        pdf_doc = PdfDocument(pdf_path)
        
        if page_num >= len(pdf_doc):
            raise ValueError(f"페이지 번호가 범위를 벗어났습니다. 총 페이지 수: {len(pdf_doc)}, 요청: {page_num}")
        
        total_pages = len(pdf_doc)
        page = pdf_doc[page_num]
        pil_image, scale = page_to_image(page, dpi=dpi)
        
        # PIL Image -> numpy array (RGB)
        img = np.array(pil_image)
        
        print(f"PDF 페이지 로드: {pdf_path}, 페이지 {page_num + 1}/{total_pages}, DPI: {dpi}, Scale: {scale:.2f}")
        
        # 리소스 정리
        page.close()
        pdf_doc.close()
        
        return img
    
    def step_by_step_preprocess(self, img: np.ndarray) -> dict:
        """
        전처리 과정을 단계별로 수행하고 중간 결과를 반환합니다.
        
        Returns:
            dict: 각 단계의 출력을 담은 딕셔너리
                - 'original': 원본 이미지
                - 'resized': 리사이즈된 이미지
                - 'normalized': 정규화된 이미지
                - 'permuted': CHW 포맷으로 변환된 이미지
                - 'batched': 배치 차원이 추가된 최종 출력
        """
        results = {}
        
        # 1. 원본 이미지
        results['original'] = img.copy()
        print(f"\n1. 원본 이미지 shape: {img.shape}, dtype: {img.dtype}")
        print(f"   값 범위: [{img.min()}, {img.max()}]")
        
        # 2. Resize
        resized = self.preprocessor.resize(img)
        results['resized'] = resized.copy()
        print(f"\n2. Resize 후 shape: {resized.shape}, dtype: {resized.dtype}")
        print(f"   값 범위: [{resized.min()}, {resized.max()}]")
        
        # 3. Normalize
        normalized = self.preprocessor.normalize(resized)
        results['normalized'] = normalized.copy()
        print(f"\n3. Normalize 후 shape: {normalized.shape}, dtype: {normalized.dtype}")
        print(f"   값 범위: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print(f"   Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        
        # 4. Permute (HWC -> CHW)
        permuted = self.preprocessor.permute(normalized)
        results['permuted'] = permuted.copy()
        print(f"\n4. Permute 후 shape: {permuted.shape}, dtype: {permuted.dtype}")
        print(f"   값 범위: [{permuted.min():.4f}, {permuted.max():.4f}]")
        
        # 5. Add Batch Dimension
        batched = np.expand_dims(permuted, axis=0).astype(np.float32)
        results['batched'] = batched
        print(f"\n5. Batch 차원 추가 후 shape: {batched.shape}, dtype: {batched.dtype}")
        print(f"   값 범위: [{batched.min():.4f}, {batched.max():.4f}]")
        
        # 6. 전체 파이프라인 한 번에 실행 (검증용)
        final_output = self.preprocessor(img)
        print(f"\n6. 전체 파이프라인 출력 shape: {final_output.shape}, dtype: {final_output.dtype}")
        print(f"   값 범위: [{final_output.min():.4f}, {final_output.max():.4f}]")
        
        # 검증: 단계별 처리와 전체 파이프라인 결과가 동일한지 확인
        if np.allclose(batched, final_output):
            print("\n✓ 검증 성공: 단계별 처리 결과와 전체 파이프라인 결과가 일치합니다.")
        else:
            print("\n✗ 검증 실패: 결과가 일치하지 않습니다!")
            diff = np.abs(batched - final_output).max()
            print(f"   최대 차이: {diff}")
        
        return results
    
    def visualize_preprocessing(self, results: dict, output_path: str = None):
        """
        전처리 단계별 결과를 시각화합니다.
        
        Args:
            results: step_by_step_preprocess()의 출력
            output_path: 저장할 이미지 경로 (None이면 화면에만 표시)
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 원본 이미지
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(results['original'])
        ax1.set_title(f"1. Original Image\nShape: {results['original'].shape}", fontsize=10)
        ax1.axis('off')
        
        # 2. Resized 이미지
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(results['resized'].astype(np.uint8))
        ax2.set_title(f"2. After Resize\nShape: {results['resized'].shape}\nRange: [0, 255]", fontsize=10)
        ax2.axis('off')
        
        # 3. Normalized 이미지 (클리핑하여 시각화)
        ax3 = fig.add_subplot(gs[0, 2])
        # 정규화된 이미지를 [0, 1] 범위로 클리핑하여 시각화
        norm_vis = np.clip(results['normalized'], 0, 1)
        ax3.imshow(norm_vis)
        ax3.set_title(f"3. After Normalize (clipped for vis)\nShape: {results['normalized'].shape}\n"
                     f"Range: [{results['normalized'].min():.2f}, {results['normalized'].max():.2f}]", 
                     fontsize=10)
        ax3.axis('off')
        
        # 4-6. Permuted 이미지의 각 채널 (CHW -> HWC로 변환하여 시각화)
        permuted_hwc = results['permuted'].transpose(1, 2, 0)  # CHW -> HWC
        
        for i, (channel_name, cmap) in enumerate([('Red', 'Reds'), ('Green', 'Greens'), ('Blue', 'Blues')]):
            ax = fig.add_subplot(gs[1, i])
            channel_data = permuted_hwc[:, :, i]
            im = ax.imshow(channel_data, cmap=cmap)
            ax.set_title(f"4. After Permute - {channel_name} Channel\n"
                        f"Range: [{channel_data.min():.2f}, {channel_data.max():.2f}]", 
                        fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 7-9. Batched 출력의 각 채널
        batched_hwc = results['batched'][0].transpose(1, 2, 0)  # (1, C, H, W) -> (H, W, C)
        
        for i, (channel_name, cmap) in enumerate([('Red', 'Reds'), ('Green', 'Greens'), ('Blue', 'Blues')]):
            ax = fig.add_subplot(gs[2, i])
            channel_data = batched_hwc[:, :, i]
            im = ax.imshow(channel_data, cmap=cmap)
            ax.set_title(f"5. Final Output - {channel_name} Channel\n"
                        f"Shape: {results['batched'].shape}\n"
                        f"Range: [{channel_data.min():.2f}, {channel_data.max():.2f}]", 
                        fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'PP-DocLayout-L Preprocessing Pipeline ({self.model_type.value})', 
                    fontsize=14, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n시각화 결과 저장: {output_path}")
        
        plt.show()
    
    def save_numpy_arrays(self, results: dict, output_dir: str):
        """
        최종 전처리 출력(batched)을 numpy 배열로 저장합니다.
        
        Args:
            results: step_by_step_preprocess()의 출력
            output_dir: 저장할 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 최종 출력(batched)만 저장
        batched_data = results['batched']
        file_path = os.path.join(output_dir, "batched.npy")
        np.save(file_path, batched_data)
        print(f"저장: {file_path} (shape: {batched_data.shape}, dtype: {batched_data.dtype})")
        
        # 메타데이터 저장
        meta_path = os.path.join(output_dir, "metadata.txt")
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Type: {self.model_type.value}\n")
            f.write(f"Input Size: {self.img_size}\n")
            f.write(f"Mean: {self.preprocessor.mean}\n")
            f.write(f"Std: {self.preprocessor.std}\n")
            f.write(f"Scale: {self.preprocessor.scale}\n\n")
            
            f.write("Saved Files:\n")
            f.write(f"  - {file_path}\n")
        
        print(f"메타데이터 저장: {meta_path}")
        return [file_path]


def main():
    parser = argparse.ArgumentParser(
        description='PP-DocLayout-L 모델의 전처리 출력을 확인합니다.'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        required=True,
        help='입력 PDF 파일 경로'
    )
    parser.add_argument(
        '--page',
        type=int,
        default=0,
        help='PDF 페이지 번호 (0부터 시작, 기본값: 0)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='PDF 렌더링 DPI (기본값: 200)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='pp_doclayout_l',
        choices=['pp_doclayout_plus_l', 'pp_doclayout_l', 'pp_doclayout_m', 'pp_doclayout_s'],
        help='모델 타입 (기본값: pp_doclayout_l)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='출력 디렉토리 (기본값: preprocessing_compare/output)'
    )
    parser.add_argument(
        '--no_vis',
        action='store_true',
        help='시각화를 생성하지 않음'
    )
    parser.add_argument(
        '--save_arrays',
        action='store_true',
        help='numpy 배열을 파일로 저장'
    )
    
    args = parser.parse_args()
    
    # PDF 경로 검증
    if not os.path.exists(args.pdf):
        print(f"오류: PDF 파일을 찾을 수 없습니다: {args.pdf}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'preprocessing_compare', 'output')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 타입 변환
    model_type_map = {
        'pp_doclayout_plus_l': ModelType.PP_DOCLAYOUT_PLUS_L,
        'pp_doclayout_l': ModelType.PP_DOCLAYOUT_L,
        'pp_doclayout_m': ModelType.PP_DOCLAYOUT_M,
        'pp_doclayout_s': ModelType.PP_DOCLAYOUT_S,
    }
    model_type = model_type_map[args.model_type]
    
    print("=" * 80)
    print("PP-DocLayout-L 전처리 출력 확인 스크립트")
    print("=" * 80)
    
    # Visualizer 초기화
    visualizer = PreprocessingVisualizer(model_type=model_type)
    
    # PDF에서 이미지 로드
    print(f"\nPDF 로드: {args.pdf}")
    print(f"페이지: {args.page + 1}, DPI: {args.dpi}")
    img = visualizer.load_pdf_page(args.pdf, page_num=args.page, dpi=args.dpi)
    source_desc = f"{Path(args.pdf).name}_page{args.page + 1}"
    
    # 단계별 전처리 수행
    print("\n" + "=" * 80)
    print("전처리 단계별 수행")
    print("=" * 80)
    results = visualizer.step_by_step_preprocess(img)
    
    # numpy 배열 저장
    if args.save_arrays:
        print("\n" + "=" * 80)
        print("numpy 배열 저장")
        print("=" * 80)
        array_dir = os.path.join(output_dir, 'arrays')
        visualizer.save_numpy_arrays(results, array_dir)
    
    # 시각화
    if not args.no_vis:
        print("\n" + "=" * 80)
        print("시각화 생성")
        print("=" * 80)
        # 파일명에 소스 정보 포함
        safe_source = source_desc.replace('.', '_').replace('/', '_')
        vis_filename = f"preprocessing_visualization_{args.model_type}_{safe_source}.png"
        vis_path = os.path.join(output_dir, vis_filename)
        visualizer.visualize_preprocessing(results, vis_path)
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()
