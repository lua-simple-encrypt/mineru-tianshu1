#!/usr/bin/env python3
"""
模型预下载脚本 - 为 CPU/GPU 离线部署准备所有必需模型 (扁平化目录版)

功能:
1. 下载 MinerU 模型到 models/PDF-Extract-Kit-1.0
2. 下载 PaddleOCR 模型到 models/PaddleOCR-VL-1.5
3. 下载 SenseVoice 等其他模型到对应一级目录
4. 自动生成 magic-pdf.json 配置文件
5. 生成模型清单 manifest.json
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
# 模型配置 (修改：target_dir 改为一级目录，PaddleOCR 改为显式下载)
# ==============================================================================
MODELS = {
    "mineru": {
        "name": "MinerU PDF-Extract-Kit",
        "repo_id": "OpenDataLab/PDF-Extract-Kit-1.0",
        "source": "modelscope",  # 建议使用 modelscope 速度更快
        "target_dir": "PDF-Extract-Kit-1.0", # 修改：直接下载到 models/PDF-Extract-Kit-1.0
        "description": "PDF OCR and layout analysis models",
        "required": True
    },
    "paddleocr": {
        "name": "PaddleOCR-VL 1.5",
        "model_id": "OpenDataLab/PaddleOCR-VL-1.5", # 指定模型 ID
        "source": "modelscope",
        "auto_download": False,          # 修改：关闭自动下载，由脚本控制路径
        "target_dir": "PaddleOCR-VL-1.5", # 修改：直接下载到 models/PaddleOCR-VL-1.5
        "description": "PaddlePaddle Vision-Language OCR model",
        "required": True
    },
    "sensevoice": {
        "name": "SenseVoice Audio Recognition",
        "model_id": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "target_dir": "SenseVoiceSmall", # 修改：扁平化目录
        "description": "Multi-language speech recognition model",
        "required": True
    },
    "paraformer": {
        "name": "Paraformer Speaker Diarization",
        "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "target_dir": "Paraformer", # 修改：扁平化目录
        "description": "Speaker diarization and VAD model",
        "required": False
    },
    "yolo11": {
        "name": "YOLO11x Watermark Detection",
        "repo_id": "corzent/yolo11x_watermark_detection",
        "filename": "best.pt",
        "source": "huggingface",
        "target_dir": "YOLO11", # 修改：扁平化目录
        "description": "Watermark detection model for document processing",
        "required": False
    },
    "lama": {
        "name": "LaMa Watermark Inpainting",
        "auto_download": True, # LaMa 保持库内部自动处理
        "description": "Will be downloaded by simple_lama_inpainting on first use",
        "required": False
    }
}


def download_from_huggingface(repo_id, target_dir, filename=None):
    """从 HuggingFace 下载模型 (修改：使用 local_dir)"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # 配置镜像（国内加速）
        hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        if filename:
            # 下载单个文件
            logger.info(f"   Downloading file: {filename}")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),      # 修改：使用 local_dir 强制指定目录
                local_dir_use_symlinks=False,   # 修改：禁用软链接
                resume_download=True
            )
        else:
            # 下载整个仓库
            logger.info(f"   Downloading repository: {repo_id}")
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),      # 修改：使用 local_dir 强制指定目录
                local_dir_use_symlinks=False,   # 修改：禁用软链接
                resume_download=True
            )

        return path

    except ImportError:
        logger.error("   ❌ huggingface_hub not installed. Install: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None


def download_from_modelscope(model_id, target_dir):
    """从 ModelScope 下载模型 (修改：使用 local_dir)"""
    try:
        from modelscope import snapshot_download

        logger.info(f"   Downloading from ModelScope: {model_id}")
        # 修改：使用 local_dir 参数，ModelScope 会直接下载到该目录，不生成随机缓存名
        path = snapshot_download(
            model_id,
            local_dir=str(target_dir), 
            revision="master"
        )

        return path

    except ImportError:
        logger.error("   ❌ modelscope not installed. Install: pip install modelscope")
        return None
    except Exception as e:
        logger.error(f"   ❌ Download failed: {e}")
        return None


def verify_model_files(path, model_name):
    """验证模型文件完整性"""
    if not path or not Path(path).exists():
        return False

    path_obj = Path(path)

    # 检查关键文件（根据不同模型类型）
    if model_name == "mineru":
        # 兼容两种结构
        has_weights = any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.bin"))
        has_subdir = (path_obj / "models").exists()
        if not (has_weights or has_subdir):
            logger.warning(f"   ⚠️  No model files or 'models' dir found in {path}")
            return False

    elif model_name == "paddleocr":
        # PaddleOCR 检查
        has_model = any(path_obj.rglob("*.safetensors")) or any(path_obj.rglob("*.pdparams"))
        if not has_model:
             logger.warning(f"   ⚠️  No PaddleOCR model files found in {path}")
             return False

    elif model_name in ["sensevoice", "paraformer"]:
        config_file = path_obj / "configuration.json"
        if not config_file.exists():
            config_file = path_obj / "config.json"
        if not any(path_obj.iterdir()):
            logger.warning(f"   ⚠️  Directory is empty: {path}")
            return False

    elif model_name == "yolo11":
        if not list(path_obj.rglob("*.pt")):
            logger.warning(f"   ⚠️  No .pt files found in {path}")
            return False

    logger.info(f"   ✅ Model files verified")
    return True


def get_directory_size(path):
    """获取目录大小（MB）"""
    if not path or not Path(path).exists():
        return 0
    path_obj = Path(path)
    if path_obj.is_file():
        return path_obj.stat().st_size / (1024 * 1024)
    total_size = sum(f.stat().st_size for f in path_obj.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)


def check_model_exists(output_path, config, name):
    """检查模型是否已存在"""
    target_dir = output_path / config["target_dir"]
    if not target_dir.exists():
        return False, "Directory not found"
    if any(target_dir.iterdir()):
        return True, "Files found"
    return False, "Directory empty"


def generate_magic_pdf_json(output_dir):
    """生成 magic-pdf.json 配置文件 (新增)"""
    project_root = Path(output_dir).parent
    config_path = project_root / "magic-pdf.json"
    
    # 路径对应容器内的挂载点 /app/models/PDF-Extract-Kit-1.0/models
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
        logger.error(f"❌ Failed to create config file: {e}")


def main(output_dir, selected_models=None, force=False):
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 Tianshu Model Download Script")
    logger.info("=" * 60)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Output directory: {output_path}")
    if force:
        logger.info("⚠️  Force mode: Will re-download existing models")
    logger.info("")

    # 筛选要下载的模型
    models_to_download = MODELS
    if selected_models:
        selected_list = [m.strip() for m in selected_models.split(",")]
        models_to_download = {k: v for k, v in MODELS.items() if k in selected_list}
        logger.info(f"📋 Selected models: {', '.join(models_to_download.keys())}")
    else:
        logger.info(f"📋 Downloading all models ({len(MODELS)} total)")

    logger.info("")

    manifest = {
        "created": datetime.now().isoformat(),
        "platform": "cpu",
        "output_dir": str(output_path),
        "models": {},
        "total_size_mb": 0
    }

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    # 下载每个模型
    for name, config in models_to_download.items():
        logger.info(f"📦 [{name.upper()}] {config['name']}")
        logger.info(f"   {config['description']}")

        try:
            if config.get("auto_download"):
                logger.info(f"   ℹ️  {name} will be downloaded automatically by library")
                
                # 记录 auto_download 的模型到 manifest
                manifest["models"][name] = {
                    "name": config["name"],
                    "status": "auto_download",
                    "description": config["description"]
                }
                continue

            # 创建目标目录
            target = output_path / config["target_dir"]
            target.mkdir(parents=True, exist_ok=True)

            # 检查模型是否已存在
            if not force:
                exists, reason = check_model_exists(output_path, config, name)
                if exists:
                    size_mb = get_directory_size(target)
                    logger.info(f"   ✅ Already exists ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {target}")
                    manifest["models"][name] = {"status": "exists", "path": str(target), "size_mb": round(size_mb, 2)}
                    total_skipped += 1
                    logger.info("")
                    continue
                else:
                    logger.info(f"   ℹ️  Not found: {reason}")

            # 下载模型
            logger.info(f"   ⬇️  Downloading...")
            path = None
            if config["source"] == "huggingface":
                path = download_from_huggingface(
                    config.get("repo_id"), # ModelScope 兼容
                    str(target), 
                    config.get("filename")
                )
            elif config["source"] == "modelscope":
                # 优先使用 model_id，如果没有则使用 repo_id
                mid = config.get("model_id") or config.get("repo_id")
                path = download_from_modelscope(mid, str(target))

            if path:
                if verify_model_files(path, name):
                    size_mb = get_directory_size(path)
                    manifest["models"][name] = {"status": "downloaded", "path": str(path), "size_mb": round(size_mb, 2)}
                    logger.info(f"   ✅ Downloaded successfully ({size_mb:.1f} MB)")
                    logger.info(f"   📂 Path: {path}")
                    total_downloaded += 1
                else:
                    total_failed += 1
            else:
                total_failed += 1

        except Exception as e:
            logger.error(f"   ❌ Error downloading {name}: {e}")
            total_failed += 1
        logger.info("")

    # 自动生成配置文件
    generate_magic_pdf_json(output_path)

    # 保存清单
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 输出总结
    logger.info("=" * 60)
    logger.info("📊 Download Summary")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully downloaded: {total_downloaded} models")
    if total_skipped > 0:
        logger.info(f"⏭️  Skipped (already exists): {total_skipped} models")
    logger.info(f"❌ Failed: {total_failed} models")
    logger.info(f"📄 Manifest saved to: {manifest_file}")
    logger.info(f"📄 Config saved to: {output_path.parent / 'magic-pdf.json'}")
    logger.info("")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download models for Tianshu (Flat Directory)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download models to ./models
  python download_models.py --output ./models

  # Force re-download
  python download_models.py --force
        """
    )
    parser.add_argument(
        "--output",
        default="./models-offline",
        help="Output directory for downloaded models (default: ./models-offline)"
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download all models"
    )

    args = parser.parse_args()

    try:
        exit_code = main(args.output, args.models, args.force)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
