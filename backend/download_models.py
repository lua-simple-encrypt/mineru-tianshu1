#!/usr/bin/env python3
"""
模型预下载脚本 - Tianshu (Final Fix v2)

修改日志:
1. [修复] PaddleOCR-VL 使用正确的 HuggingFace 仓库: PaddlePaddle/PaddleOCR-VL-1.5
2. [保持] MinerU VLM 使用 ModelScope ID: opendatalab/MinerU2.5-2509-1.2B
3. [保持] PaddleX 模型使用 HuggingFace 源
4. [保持] 目录结构扁平化
"""

import os
import sys
import json
import tarfile
import shutil
import requests
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
from tqdm import tqdm

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ==============================================================================
# 模型配置
# ==============================================================================
MODELS = {
    # 1. MinerU Pipeline
    "mineru_pipeline": {
        "name": "MinerU Pipeline (PDF-Extract-Kit)",
        "repo_id": "OpenDataLab/PDF-Extract-Kit-1.0",
        "source": "modelscope",
        "target_dir": "PDF-Extract-Kit-1.0",
        "description": "PDF OCR, Layout Analysis models",
        "required": True
    },
    # 2. MinerU VLM
    "mineru_vlm": {
        "name": "MinerU 2.5 VLM (1.2B)",
        "model_id": "opendatalab/MinerU2.5-2509-1.2B", 
        "source": "modelscope",
        "target_dir": "MinerU2.5-2509-1.2B", 
        "description": "Vision Language Model for high-precision parsing",
        "required": True
    },
    # 3. PaddleOCR-VL (修复为正确的 HF 仓库)
    "paddleocr": {
        "name": "PaddleOCR-VL 1.5",
        "repo_id": "PaddlePaddle/PaddleOCR-VL-1.5", # 【修正】使用您提供的正确地址
        "source": "huggingface", 
        "target_dir": "PaddleOCR-VL-1.5",
        "description": "PaddlePaddle Vision-Language OCR model",
        "required": True
    },
    # 4. PaddleX DocLayout
    "pp_layout": {
        "name": "PP-DocLayoutV3",
        "repo_id": "PaddlePaddle/PP-DocLayoutV3",
        "source": "huggingface", 
        "target_dir": "PP-DocLayoutV3",
        "description": "PaddleX Document Layout Analysis Model",
        "required": True
    },
    # 5. PaddleX Orientation
    "pp_lcnet": {
        "name": "PP-LCNet Doc Orientation",
        "repo_id": "PaddlePaddle/PP-LCNet_x1_0_doc_ori",
        "source": "huggingface",
        "target_dir": "PP-LCNet_x1_0_doc_ori",
        "description": "PaddleX Document Orientation Classification Model",
        "required": True
    },
    # 6. SenseVoice
    "sensevoice": {
        "name": "SenseVoice Audio Recognition",
        "model_id": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "target_dir": "SenseVoiceSmall",
        "required": True
    },
    # 7. Paraformer
    "paraformer": {
        "name": "Paraformer Speaker Diarization",
        "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "target_dir": "Paraformer",
        "required": False
    },
    # 8. YOLO11 Watermark
    "yolo11": {
        "name": "YOLO11x Watermark Detection",
        "repo_id": "corzent/yolo11x_watermark_detection",
        "filename": "best.pt",
        "source": "huggingface",
        "target_dir": "YOLO11",
        "required": False
    },
    "lama": {
        "name": "LaMa Watermark Inpainting",
        "auto_download": True,
        "description": "Will be downloaded by simple_lama_inpainting on first use",
        "required": False
    }
}

# --- 下载函数 ---

def download_from_huggingface(repo_id, target_dir, filename=None):
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        
        if filename:
            logger.info(f"   Downloading file: {filename}")
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir), local_dir_use_symlinks=False, resume_download=True)
        else:
            logger.info(f"   Downloading repository: {repo_id}")
            path = snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False, resume_download=True)
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

def download_from_modelscope(model_id, target_dir):
    try:
        from modelscope import snapshot_download
        logger.info(f"   Downloading from ModelScope: {model_id}")
        path = snapshot_download(model_id, local_dir=str(target_dir), revision="master")
        return path
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None

def download_url_tar_multi(urls, target_dir):
    """下载多个 tar 包并解压到同一目录"""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    success = True
    for url in urls:
        filename = url.split('/')[-1]
        try:
            logger.info(f"   ⬇️  Fetching: {filename}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                logger.error(f"   ❌ HTTP Error {response.status_code}")
                success = False
                continue
                
            tar_path = target_path / filename
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f, tqdm(desc=filename, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
            
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=target_path)
                for member in tar.getnames():
                    member_path = target_path / member
                    if member_path.is_dir():
                        for subfile in member_path.iterdir():
                            shutil.move(str(subfile), str(target_path / subfile.name))
                        if not any(member_path.iterdir()):
                            member_path.rmdir()
            tar_path.unlink()
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            success = False
            
    return str(target_path) if success else None

# --- 验证与辅助 ---

def verify_model_files(path, model_name):
    path_obj = Path(path)
    if not path_obj.exists(): return False

    if model_name == "mineru_pipeline":
        if not (any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.bin"))):
            if (path_obj / "models").exists(): return True
            logger.warning(f"   ⚠️  No model files in {path}")
            return False
    elif model_name == "mineru_vlm":
        if not any(path_obj.rglob("*.safetensors")):
            logger.warning(f"   ⚠️  No safetensors found in {path}")
            return False
    elif model_name in ["paddleocr", "pp_layout", "pp_lcnet"]:
        # PaddleOCR-VL 可能只包含 safetensors，也可能包含 pdiparams，放宽检查
        if not (any(path_obj.rglob("*.pdiparams")) or any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.bin"))):
             logger.warning(f"   ⚠️  No model weights found in {path}")
             return False
    elif model_name == "yolo11":
        if path_obj.is_file(): return path_obj.suffix == ".pt"
        if not list(path_obj.rglob("*.pt")):
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
    project_root = Path(output_dir).parent
    config_path = project_root / "magic-pdf.json"
    
    config_content = r"""{
  "models-dir": "/app/models/PDF-Extract-Kit-1.0/models",
  "device-mode": "cuda",
  "layout-config": {
    "model": "layoutlmv3",
    "batch_size": 2
  },
  "formula-config": {
    "mfd_model": "yolo_v8",
    "mre_model": "unimernet",
    "batch_size": 2
  }
}"""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        logger.success(f"✅ Configuration file created at: {config_path}")
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")

def main(output_dir, selected_models=None, force=False):
    logger.info("=" * 60)
    logger.info("🚀 Tianshu Model Download Script (Corrected)")
    logger.info("=" * 60)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    models_to_download = MODELS
    if selected_models:
        selected_list = [m.strip() for m in selected_models.split(",")]
        models_to_download = {k: v for k, v in MODELS.items() if k in selected_list}

    manifest = {"created": datetime.now().isoformat(), "models": {}, "total_size_mb": 0}
    total_dl, total_skip, total_fail = 0, 0, 0

    for name, config in models_to_download.items():
        logger.info(f"📦 [{name.upper()}] {config['name']}")
        
        try:
            target = output_path / config["target_dir"]
            
            if not force:
                exists, reason = check_model_exists(output_path, config, name)
                if exists:
                    size_mb = get_directory_size(target)
                    logger.info(f"   ✅ Already exists ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {target}")
                    manifest["models"][name] = {"status": "exists", "path": str(target)}
                    total_skip += 1
                    logger.info("")
                    continue

            logger.info(f"   ⬇️  Downloading...")
            path = None
            src = config["source"]
            
            if src == "huggingface":
                path = download_from_huggingface(config["repo_id"], str(target), config.get("filename"))
            elif src == "modelscope":
                mid = config.get("model_id") or config.get("repo_id")
                path = download_from_modelscope(mid, str(target))
            elif src == "url_tar_multi":
                path = download_url_tar_multi(config["urls"], str(target))
            elif src == "url_tar":
                path = download_and_extract_tar(config["url"], str(target))

            if path and verify_model_files(path, name):
                size_mb = get_directory_size(path)
                manifest["models"][name] = {"status": "downloaded", "path": str(path)}
                logger.info(f"   ✅ Success ({size_mb:.1f} MB)")
                total_dl += 1
            else:
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
