"""
PaddleOCR-VL 解析引擎 (PaddleX v3 Wrapper)
单例模式，每个进程只加载一次模型
支持自动多语言识别、Markdown 格式输出

参考文档：http://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
"""

import os
import sys
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger

# 尝试导入 paddle 和 paddlex
try:
    import paddle
    from paddlex import create_pipeline
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("⚠️ PaddlePaddle or PaddleX not installed. Please install: pip install paddlepaddle-gpu paddlex")

class PaddleOCRVLEngine:
    """
    PaddleOCR-VL 解析引擎（基于 PaddleX v3）

    特性：
    - 单例模式：确保进程内只有一个模型实例
    - 显存管理：支持推理后清理显存
    - 格式支持：输出 Markdown 和 JSON
    - 参数支持：支持 PaddleOCR-VL-1.5 的全量参数配置
    - 内存优化：使用生成器流式处理长文档，防止 OOM
    """

    _instance: Optional["PaddleOCRVLEngine"] = None
    _lock = Lock()
    _pipeline = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0", model_name: str = "PaddleOCR-VL-1.5-0.9B"):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.model_name = model_name
            self.gpu_id = 0

            # 解析 GPU ID
            if "cuda" in device.lower():
                try:
                    parts = device.split(":")
                    if len(parts) > 1:
                        self.gpu_id = int(parts[-1])
                except ValueError:
                    logger.warning(f"⚠️ Invalid device format '{device}', defaulting to GPU 0")

            self._check_environment()
            
            self._initialized = True
            logger.info(f"🔧 PaddleOCR-VL Engine initialized (Model: {self.model_name}, Device: {self.device})")

    def _check_environment(self):
        """检查 GPU 和 Paddle 环境"""
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddlePaddle environment is missing.")

        if not paddle.device.is_compiled_with_cuda():
            logger.error("❌ PaddlePaddle is installed but NOT compiled with CUDA.")
            raise RuntimeError("PaddlePaddle CUDA version required.")

        try:
            gpu_name = paddle.device.cuda.get_device_name(self.gpu_id)
            logger.info(f"✅ GPU Detected: {gpu_name}")
        except Exception:
            pass

    def _load_pipeline(self):
        """延迟加载 PaddleX Pipeline"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info(f"📥 Loading PaddleOCR-VL Pipeline: {self.model_name}...")
            start_time = time.time()

            # 设置设备
            paddle.set_device(f"gpu:{self.gpu_id}")

            # 确定模型路径 (优先使用本地缓存)
            pipeline_source = self.model_name
            local_base = Path("/app/models/paddlex") / self.model_name
            pdx_home = os.environ.get("PADDLEX_HOME")

            if local_base.exists() and any(local_base.iterdir()):
                logger.info(f"📂 Using local model: {local_base}")
                pipeline_source = str(local_base)
            elif pdx_home:
                logger.info(f"💾 Using PADDLEX_HOME: {pdx_home}")
            
            try:
                # 创建 Pipeline
                self._pipeline = create_pipeline(
                    pipeline=pipeline_source,
                    device=f"gpu:{self.gpu_id}",
                    use_hpip=False 
                )
                
                logger.success(f"✅ Pipeline loaded in {time.time() - start_time:.2f}s")
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Failed to load pipeline: {e}")
                raise RuntimeError(f"PaddleOCR-VL load failed: {e}")

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        执行解析

        Args:
            file_path: 输入文件路径
            output_path: 输出目录
            **kwargs: 支持 PaddleOCR-VL 的所有参数 (支持驼峰或下划线命名)
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🤖 Processing: {file_path.name}")
        
        pipeline = self._load_pipeline()
        
        # 参数映射表 (API 驼峰 -> PaddleX 下划线)
        param_mapping = {
            # 功能开关
            "useDocOrientationClassify": "use_doc_orientation_classify",
            "useDocUnwarping": "use_doc_unwarping",
            "useLayoutDetection": "use_layout_parsing",
            "useChartRecognition": "use_chart_recognition",
            "useSealRecognition": "use_seal_recognition",
            "useOcrForImageBlock": "use_ocr_for_image_block",
            "layoutNms": "layout_nms",
            # 后处理参数
            "markdownIgnoreLabels": "markdown_ignore_labels",
            "mergeTables": "merge_tables",
            "relevelTitles": "relevel_titles",
            "restructurePages": "restructure_pages",
            "layoutShapeMode": "layout_shape_mode",
            "minPixels": "min_pixels",
            "maxPixels": "max_pixels",
            # 生成参数
            "promptLabel": "prompt_label", 
            "temperature": "temperature",
            "topP": "top_p",
            "repetitionPenalty": "repetition_penalty"
        }

        # 1. 规范化参数并过滤天枢其他无关参数
        # (只传递被 param_mapping 记录的 PaddleX 参数，防止 predict 报错 TypeError: unexpected keyword argument)
        predict_params = {}
        for k, v in kwargs.items():
            if k in param_mapping:
                predict_params[param_mapping[k]] = v

        try:
            # =================================================================
            # 动态检查 pipeline 是否具备预处理能力
            # =================================================================
            has_preprocessor = hasattr(pipeline, "doc_preprocessor_pipeline") and pipeline.doc_preprocessor_pipeline is not None
            
            req_orientation = predict_params.get("use_doc_orientation_classify", False)
            req_unwarping = predict_params.get("use_doc_unwarping", False)

            # 如果请求了预处理功能但模型不支持，强制关闭并警告
            if (req_orientation or req_unwarping) and not has_preprocessor:
                logger.warning("⚠️ 请求了文档矫正/分类，但模型缺少预处理模块。已自动禁用以防止崩溃。")
                predict_params["use_doc_orientation_classify"] = False
                predict_params["use_doc_unwarping"] = False
            
            # 默认参数兜底
            if "use_layout_parsing" not in predict_params:
                predict_params["use_layout_parsing"] = True
            if "use_doc_orientation_classify" not in predict_params:
                predict_params["use_doc_orientation_classify"] = False
            if "use_doc_unwarping" not in predict_params:
                predict_params["use_doc_unwarping"] = False

            # 设置输入文件
            predict_params["input"] = str(file_path)

            # 打印最终参数 (排除 input 以防日志过长)
            log_params = {k: v for k, v in predict_params.items() if k != "input"}
            logger.info(f"🚀 开始推理 (参数: {json.dumps(log_params, default=str, ensure_ascii=False)})")
            
            # 执行推理
            # 【性能优化】不使用 list(output) 全部加载到内存，改为生成器流式处理，防止长 PDF 导致 OOM
            output_generator = pipeline.predict(**predict_params)
            
            markdown_pages = []
            page_count = 0
            
            for res in output_generator:
                page_count += 1
                page_dir = output_path / f"page_{page_count}"
                page_dir.mkdir(parents=True, exist_ok=True)

                # 保存图片和JSON
                if hasattr(res, "save_to_img"): res.save_to_img(str(page_dir))
                if hasattr(res, "save_to_json"): res.save_to_json(str(page_dir))

                # 提取 Markdown
                page_md = ""
                
                # 兼容 PaddleX 不同的 Markdown 存储结构
                if hasattr(res, "markdown") and res.markdown:
                    if isinstance(res.markdown, dict):
                        page_md = res.markdown.get('markdown_texts', '')
                        if not page_md:
                            page_md = res.markdown.get('text', '')
                    elif hasattr(res.markdown, 'markdown_texts'):
                        page_md = res.markdown.markdown_texts
                    elif isinstance(res.markdown, str):
                        page_md = res.markdown
                    else:
                        page_md = str(res.markdown)
                
                elif hasattr(res, "str") and res.str:
                    page_md = str(res.str)
                
                # 尝试从保存的文件读取（最可靠的方式兜底）
                if not page_md and hasattr(res, "save_to_markdown"):
                    try:
                        res.save_to_markdown(str(page_dir))
                        saved_mds = list(page_dir.glob("*.md"))
                        if saved_mds:
                            page_md = saved_mds[0].read_text(encoding="utf-8")
                    except Exception:
                        pass

                if page_md:
                    markdown_pages.append(page_md)
                else:
                    logger.warning(f"⚠️ Page {page_count}: No markdown content extracted.")

            logger.info(f"📄 Successfully processed {page_count} pages")

            # 合并结果
            full_markdown = "\n\n---\n\n".join(markdown_pages)
            final_md_path = output_path / "result.md"
            final_md_path.write_text(full_markdown, encoding="utf-8")
            
            return {
                "success": True,
                "result_path": str(output_path),
                "markdown": full_markdown,
                "markdown_file": str(final_md_path)
            }

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """清理显存"""
        if PADDLE_AVAILABLE and paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
            gc.collect()

# 全局单例
_engine_instance = None

def get_engine(model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PaddleOCRVLEngine(model_name=model_name)
    return _engine_instance
