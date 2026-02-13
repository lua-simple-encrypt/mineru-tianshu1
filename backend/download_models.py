#!/usr/bin/env python3
"""
模型预下载脚本 - Tianshu (Official 3-Options Support + Full PaddleX Models)

支持官方的三种解析引擎选项:
1. pipeline (传统多模型管道)
2. vlm-auto-engine (VLM 自动引擎)
3. hybrid-auto-engine (混合高精度引擎)

同时下载所有指定的 PaddleX/PaddleOCR 模型到 /app/models/paddlex/ 目录下
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ==============================================================================
# 模型配置清单
# ==============================================================================
MODELS = {
    # -------------------------------------------------------------------------
    # 1. MinerU 核心模型 (保持扁平结构)
    # -------------------------------------------------------------------------
    "mineru_pipeline": {
        "name": "MinerU Pipeline (PDF-Extract-Kit)",
        "repo_id": "OpenDataLab/PDF-Extract-Kit-1.0",
        "source": "modelscope",
        "target_dir": "PDF-Extract-Kit-1.0",
        "description": "PDF OCR, Layout Analysis models (For 'pipeline' mode)",
        "required": True
    },
    "mineru_vlm": {
        "name": "MinerU 2.5 VLM (1.2B)",
        "model_id": "opendatalab/MinerU2.5-2509-1.2B",
        "source": "modelscope",
        "target_dir": "MinerU2.5-2509-1.2B",
        "description": "Vision Language Model (For 'vlm-auto-engine' & 'hybrid-auto-engine')",
        "required": True
    },

    # -------------------------------------------------------------------------
    # 2. PaddleX / PaddleOCR 模型 (全部归档到 paddlex/ 子目录)
    # -------------------------------------------------------------------------
    
    # --- 多模态文档解析 ---
    "paddleocr_vl_1_5": {
        "name": "PaddleOCR-VL-1.5-0.9B",
        "repo_id": "PaddlePaddle/PaddleOCR-VL-1.5",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PaddleOCR-VL-1.5-0.9B",
        "target_dir": "paddlex/PaddleOCR-VL-1.5-0.9B",
        "description": "多模态文档解析模型 v1.5",
        "required": True
    },
    "paddleocr_vl_0_9": {
        "name": "PaddleOCR-VL-0.9B",
        "repo_id": "PaddlePaddle/PaddleOCR-VL-0.9B",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PaddleOCR-VL-0.9B",
        "target_dir": "paddlex/PaddleOCR-VL-0.9B",
        "description": "多模态文档解析模型 v1.0",
        "required": False
    },

    # --- 版面分析 (Layout) ---
    "pp_doclayout_v3": {
        "name": "PP-DocLayoutV3",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-DocLayoutV3",
        "target_dir": "paddlex/PP-DocLayoutV3",
        "required": True
    },
    "pp_doclayout_v2": {
        "name": "PP-DocLayoutV2",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-DocLayoutV2",
        "target_dir": "paddlex/PP-DocLayoutV2",
        "required": False
    },
    "pp_doclayout_plus_l": {
        "name": "PP-DocLayout_plus-L",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-DocLayout_plus-L",
        "target_dir": "paddlex/PP-DocLayout_plus-L",
        "required": False
    },
    "pp_docblocklayout": {
        "name": "PP-DocBlockLayout",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-DocBlockLayout",
        "target_dir": "paddlex/PP-DocBlockLayout",
        "required": False
    },

    # --- 文档矫正/方向分类 ---
    "pp_lcnet_doc_ori": {
        "name": "PP-LCNet_x1_0_doc_ori",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-LCNet_x1_0_doc_ori",
        "target_dir": "paddlex/PP-LCNet_x1_0_doc_ori",
        "required": True
    },
    "pp_lcnet_textline_ori": {
        "name": "PP-LCNet_x1_0_textline_ori",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-LCNet_x1_0_textline_ori",
        "target_dir": "paddlex/PP-LCNet_x1_0_textline_ori",
        "required": False
    },
    "pp_lcnet_x0_25_textline_ori": {
        "name": "PP-LCNet_x0_25_textline_ori",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-LCNet_x0_25_textline_ori",
        "target_dir": "paddlex/PP-LCNet_x0_25_textline_ori",
        "required": False
    },
    "uvdoc": {
        "name": "UVDoc (Doc Unwarping)",
        "source": "modelscope",
        "model_id": "PaddlePaddle/UVDoc",
        "target_dir": "paddlex/UVDoc",
        "required": False
    },

    # --- 通用 OCR (PP-OCRv5) ---
    "pp_ocrv5_det": {
        "name": "PP-OCRv5_mobile_det",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-OCRv5_mobile_det",
        "target_dir": "paddlex/PP-OCRv5_mobile_det",
        "required": False
    },
    "pp_ocrv5_rec": {
        "name": "PP-OCRv5_mobile_rec",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-OCRv5_mobile_rec",
        "target_dir": "paddlex/PP-OCRv5_mobile_rec",
        "required": False
    },
    "pp_ocrv5_server_rec": {
        "name": "PP-OCRv5_server_rec",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-OCRv5_server_rec",
        "target_dir": "paddlex/PP-OCRv5_server_rec",
        "required": False
    },
    "pp_ocrv4_server_seal_det": {
        "name": "PP-OCRv4_server_seal_det",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-OCRv4_server_seal_det",
        "target_dir": "paddlex/PP-OCRv4_server_seal_det",
        "required": False
    },

    # --- 多语言 OCR ---
    "eslav_pp_ocrv5_mobile_rec": {
        "name": "eslav_PP-OCRv5_mobile_rec",
        "source": "modelscope",
        "model_id": "PaddlePaddle/eslav_PP-OCRv5_mobile_rec",
        "target_dir": "paddlex/eslav_PP-OCRv5_mobile_rec",
        "required": False
    },
    "korean_pp_ocrv5_mobile_rec": {
        "name": "korean_PP-OCRv5_mobile_rec",
        "source": "modelscope",
        "model_id": "PaddlePaddle/korean_PP-OCRv5_mobile_rec",
        "target_dir": "paddlex/korean_PP-OCRv5_mobile_rec",
        "required": False
    },
    "latin_pp_ocrv5_mobile_rec": {
        "name": "latin_PP-OCRv5_mobile_rec",
        "source": "modelscope",
        "model_id": "PaddlePaddle/latin_PP-OCRv5_mobile_rec",
        "target_dir": "paddlex/latin_PP-OCRv5_mobile_rec",
        "required": False
    },

    # --- 公式/表格识别 ---
    "pp_formulanet": {
        "name": "PP-FormulaNet_plus-L",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-FormulaNet_plus-L",
        "target_dir": "paddlex/PP-FormulaNet_plus-L",
        "required": False
    },
    "pp_lcnet_table_cls": {
        "name": "PP-LCNet_x1_0_table_cls",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-LCNet_x1_0_table_cls",
        "target_dir": "paddlex/PP-LCNet_x1_0_table_cls",
        "required": False
    },
    "pp_chart2table": {
        "name": "PP-Chart2Table",
        "source": "modelscope",
        "model_id": "PaddlePaddle/PP-Chart2Table",
        "target_dir": "paddlex/PP-Chart2Table",
        "required": False
    },
    "slanext_wired": {
        "name": "SLANeXt_wired",
        "source": "modelscope",
        "model_id": "PaddlePaddle/SLANeXt_wired",
        "target_dir": "paddlex/SLANeXt_wired",
        "required": False
    },
    "slanet_plus": {
        "name": "SLANet_plus",
        "source": "modelscope",
        "model_id": "PaddlePaddle/SLANet_plus",
        "target_dir": "paddlex/SLANet_plus",
        "required": False
    },
    "rtdetr_wired": {
        "name": "RT-DETR-L_wired_table_cell_det",
        "source": "modelscope",
        "model_id": "PaddlePaddle/RT-DETR-L_wired_table_cell_det",
        "target_dir": "paddlex/RT-DETR-L_wired_table_cell_det",
        "required": False
    },
    "rtdetr_wireless": {
        "name": "RT-DETR-L_wireless_table_cell_det",
        "source": "modelscope",
        "model_id": "PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
        "target_dir": "paddlex/RT-DETR-L_wireless_table_cell_det",
        "required": False
    },

    # -------------------------------------------------------------------------
    # 3. 其他模型 (Audio / Image)
    # -------------------------------------------------------------------------
    "sensevoice": {
        "name": "SenseVoice Audio Recognition",
        "model_id": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "target_dir": "SenseVoiceSmall",
        "description": "Multi-language speech recognition model",
        "required": True
    },
    "paraformer": {
        "name": "Paraformer Speaker Diarization",
        "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "target_dir": "Paraformer",
        "description": "Speaker diarization and VAD model",
        "required": False
    },
    "yolo11": {
        "name": "YOLO11x Watermark Detection",
        "repo_id": "corzent/yolo11x_watermark_detection",
        "filename": "best.pt",
        "source": "huggingface",
        "target_dir": "YOLO11",
        "description": "Watermark detection model",
        "required": False
    },
    "lama": {
        "name": "LaMa Watermark Inpainting",
        "auto_download": True,
        "description": "Will be downloaded by simple_lama_inpainting on first use",
        "required": False
    }
}

# ==============================================================================
# 下载函数 (保持不变)
# ==============================================================================

def download_from_huggingface(repo_id, target_dir, filename=None):
    """从 HuggingFace 下载"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        
        # 配置国内镜像
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        
        if filename:
            logger.info(f"   Downloading file: {filename}")
            path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                local_dir=str(target_dir), 
                local_dir_use_symlinks=False, 
                resume_download=True
            )
        else:
            logger.info(f"   Downloading repository: {repo_id}")
            path = snapshot_download(
                repo_id=repo_id, 
                local_dir=str(target_dir), 
                local_dir_use_symlinks=False, 
                resume_download=True
            )
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

def download_from_modelscope(model_id, target_dir):
    """从 ModelScope 下载"""
    try:
        from modelscope import snapshot_download
        
        logger.info(f"   Downloading from ModelScope: {model_id}")
        path = snapshot_download(
            model_id, 
            local_dir=str(target_dir), 
            revision="master"
        )
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

# ==============================================================================
# 验证与辅助函数
# ==============================================================================

def verify_model_files(path, model_name):
    """验证下载是否完整"""
    path_obj = Path(path)
    if not path_obj.exists(): return False

    # 1. MinerU Pipeline
    if model_name == "mineru_pipeline":
        if not (any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.bin"))):
            if (path_obj / "models").exists(): return True
            logger.warning(f"   ⚠️  No model files in {path}")
            return False
            
    # 2. MinerU VLM
    elif model_name == "mineru_vlm":
        if not any(path_obj.rglob("*.safetensors")):
            logger.warning(f"   ⚠️  No safetensors found in {path}")
            return False
            
    # 3. Paddle Models (OCR, Layout, LCNet)
    elif "paddle" in model_name or "pp_" in model_name or "slanext" in model_name or "uvdoc" in model_name or "rtdetr" in model_name:
         # PaddleX 模型通常包含 inference.pdmodel 等文件
         if not (any(path_obj.rglob("*.pdmodel")) or any(path_obj.rglob("*.pdiparams")) or any(path_obj.rglob("*.yaml"))):
              logger.warning(f"   ⚠️  No Paddle inference files found in {path}")
              return False
              
    # 4. YOLO (单文件或目录)
    elif model_name == "yolo11":
        if path_obj.is_file():
            if path_obj.suffix != ".pt":
                return False
        elif not list(path_obj.rglob("*.pt")):
            logger.warning(f"   ⚠️  No .pt files found")
            return False
            
    logger.info(f"   ✅ Model files verified")
    return True

def get_directory_size(path):
    path_obj = Path(path)
    if not path_obj.exists(): return 0
    if path_obj.is_file(): return path_obj.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path_obj.rglob("*") if f.is_file()) / (1024 * 1024)

def check_model_exists(output_path, config, name):
    target_dir = output_path / config["target_dir"]
    if config.get("filename"):
        f = target_dir / config["filename"]
        return (f.exists() and f.stat().st_size > 0), "File found"
    if not target_dir.exists(): return False, "Dir missing"
    if any(target_dir.iterdir()): return True, "Files found"
    return False, "Dir empty"

def generate_magic_pdf_json(output_dir):
    """生成 magic-pdf.json，同时配置 Pipeline 和 VLM"""
    project_root = Path(output_dir).parent
    config_path = project_root / "magic-pdf.json"
    
    # 注意：这里的路径是 Docker 容器内的路径
    # models-dir 指向 MinerU Pipeline 的 models 子目录
    config_content = r"""{
  "models-dir": "/app/models/PDF-Extract-Kit-1.0/models",
  "vlm-models-dir": "/app/models/MinerU2.5-2509-1.2B",
  "device-mode": "cuda",
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "formula-config": {
    "mfd_model": "yolo_v8_mfd",
    "mre_model": "unimernet_small"
  }
}"""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        logger.success(f"✅ Configuration file created at: {config_path}")
        logger.info("   -> Confirmed support for: pipeline, vlm-auto-engine, hybrid-auto-engine")
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")

# ==============================================================================
# 主程序
# ==============================================================================

def main(output_dir, selected_models=None, force=False):
    logger.info("=" * 60)
    logger.info("🚀 Tianshu Model Download Script (Official 3-Options + PaddleX)")
    logger.info("=" * 60)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    # 筛选模型
    models_to_download = MODELS
    if selected_models:
        selected_list = [m.strip() for m in selected_models.split(",")]
        models_to_download = {k: v for k, v in MODELS.items() if k in selected_list}

    manifest = {"created": datetime.now().isoformat(), "models": {}, "total_size_mb": 0}
    total_dl, total_skip, total_fail = 0, 0, 0

    for name, config in models_to_download.items():
        logger.info(f"📦 [{name.upper()}] {config['name']}")
        
        try:
            # 自动下载模型跳过
            if config.get("auto_download"):
                logger.info(f"   ℹ️  {name} will be auto-downloaded by library")
                manifest["models"][name] = {"status": "auto_download"}
                continue

            target = output_path / config["target_dir"]
            
            # 检查存在
            if not force:
                exists, reason = check_model_exists(output_path, config, name)
                if exists:
                    size_mb = get_directory_size(target)
                    logger.info(f"   ✅ Already exists ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {target}")
                    manifest["models"][name] = {"status": "exists", "path": str(target), "size_mb": round(size_mb, 2)}
                    total_skip += 1
                    logger.info("")
                    continue

            # 下载
            logger.info(f"   ⬇️  Downloading to {config['target_dir']}...")
            path = None
            src = config["source"]
            
            if src == "huggingface":
                path = download_from_huggingface(
                    config["repo_id"], 
                    str(target), 
                    config.get("filename")
                )
            elif src == "modelscope":
                # 优先使用 model_id，如果没有则用 repo_id (兼容旧配置)
                mid = config.get("model_id") or config.get("repo_id")
                path = download_from_modelscope(mid, str(target))

            # 验证
            if path and verify_model_files(path, name):
                size_mb = get_directory_size(path)
                manifest["models"][name] = {"status": "downloaded", "path": str(path)}
                logger.info(f"   ✅ Success ({size_mb:.1f} MB)")
                logger.info(f"   📂 Path: {path}")
                total_dl += 1
            else:
                logger.error(f"   ❌ Validation failed for {name}")
                total_fail += 1

        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            total_fail += 1
        logger.info("")

    generate_magic_pdf_json(output_path)
    
    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ Downloaded: {total_dl} | ⏭️  Skipped: {total_skip} | ❌ Failed: {total_fail}")
    return 0 if total_fail == 0 else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./models")
    parser.add_argument("--models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    try:
        sys.exit(main(args.output, args.models, args.force))
    except KeyboardInterrupt:
        sys.exit(130)
