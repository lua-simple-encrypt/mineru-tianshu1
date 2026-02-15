"""
PaddleOCR-VL 解析引擎 (PaddleX v3 Local Wrapper)
单例模式，每个进程只加载一次模型
支持自动多语言识别、Markdown 格式输出

优化日志：
1. 强制单线程 (PARALLEL_WORKER_NUM=1) 以解决 "Already borrowed" 竞态崩溃
2. 禁用模型源联网检查 (DISABLE_MODEL_SOURCE_CHECK) 加快启动
3. 增加结果处理的防御性检查 (NoneType protection)
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

# ==============================================================================
# 🚨 全局环境配置 (必须在导入 paddlex 之前设置)
# ==============================================================================
# 1. 限制 PaddleX 内部推理并发数为 1，防止高并发请求导致显存溢出或底层 C++ 竞态冲突
os.environ["PADDLEX_INFERENCE_PARALLEL_WORKER_NUM"] = "1"
# 2. 禁用模型源检查，加快启动速度 (内网/无网环境必备)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# ==============================================================================

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
    PaddleOCR-VL 解析引擎（基于 PaddleX v3 本地推理）

    特性：
    - 单例模式：确保进程内只有一个模型实例
    - 显存管理：支持推理后清理显存
    - 格式支持：输出 Markdown 和 JSON
    - 稳定性：强制串行处理，防止 OOM
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
                    self.gpu_id = 0
                    logger.warning(f"⚠️ Invalid device format '{device}', defaulting to GPU 0")

            self._check_environment()
            
            self._initialized = True
            logger.info(f"🔧 PaddleOCR-VL Local Engine initialized (Model: {self.model_name}, Device: GPU {self.gpu_id})")

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

            # 确定模型路径 (优先使用 PADDLEX_HOME 缓存)
            # 默认官方模型路径通常在 ~/.paddlex/official_models/
            pdx_home = os.environ.get("PADDLEX_HOME", "/root/.paddlex")
            local_cache_path = Path(pdx_home) / "official_models" / self.model_name
            
            pipeline_source = self.model_name # 默认为模型名称，让 PaddleX 自动处理

            # 如果本地存在模型文件，优先使用本地路径，避免尝试下载
            if local_cache_path.exists() and any(local_cache_path.iterdir()):
                logger.info(f"📂 Found local model cache: {local_cache_path}")
                pipeline_source = str(local_cache_path)
            else:
                logger.info(f"🌐 Local model not found at {local_cache_path}, attempting auto-download...")
            
            try:
                # 创建 Pipeline
                self._pipeline = create_pipeline(
                    pipeline=pipeline_source,
                    device=f"gpu:{self.gpu_id}",
                    use_hpip=False # 禁用高性能推理代理，以获得更好兼容性
                )
                
                logger.success(f"✅ Pipeline loaded in {time.time() - start_time:.2f}s")
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Failed to load pipeline: {e}")
                logger.error(traceback.format_exc())
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
            "useDocOrientationClassify": "use_doc_orientation_classify",
            "useDocUnwarping": "use_doc_unwarping",
            "useLayoutDetection": "use_layout_parsing",
            "useChartRecognition": "use_chart_recognition",
            "useSealRecognition": "use_seal_recognition",
            "useOcrForImageBlock": "use_ocr_for_image_block",
            "layoutNms": "layout_nms",
            "markdownIgnoreLabels": "markdown_ignore_labels",
            "mergeTables": "merge_tables",
            "relevelTitles": "relevel_titles",
            "restructurePages": "restructure_pages",
            "layoutShapeMode": "layout_shape_mode",
            "minPixels": "min_pixels",
            "maxPixels": "max_pixels",
        }

        # 1. 规范化参数并过滤无关参数
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
            if "use_seal_recognition" not in predict_params:
                predict_params["use_seal_recognition"] = True # 默认开启印章识别

            # 设置输入文件
            predict_params["input"] = str(file_path)

            log_params = {k: v for k, v in predict_params.items() if k != "input"}
            logger.info(f"🚀 开始推理 (参数: {json.dumps(log_params, default=str, ensure_ascii=False)})")
            
            # 执行推理 (流式生成器)
            # 注意：由于我们在文件头设置了 PARALLEL_WORKER_NUM=1，这里会串行处理
            output_generator = pipeline.predict(**predict_params)
            
            markdown_pages = []
            json_list = []
            page_count = 0
            
            for res in output_generator:
                page_count += 1
                
                # 🛡️ 关键防御：防止 None 结果导致崩溃
                if res is None:
                    logger.error(f"❌ Page {page_count} returned None result")
                    continue

                page_dir = output_path / f"page_{page_count}"
                page_dir.mkdir(parents=True, exist_ok=True)

                # 1. 保存图片和JSON
                try:
                    if hasattr(res, "save_to_img"): res.save_to_img(str(page_dir))
                    if hasattr(res, "save_to_json"): res.save_to_json(str(page_dir))
                except Exception as e:
                    logger.warning(f"⚠️ Page {page_count}: Failed to save assets: {e}")

                # 2. 收集 JSON 数据
                if hasattr(res, "json") and res.json:
                    json_list.append(res.json)

                # 3. 提取 Markdown (增强兼容性)
                page_md = ""
                try:
                    if hasattr(res, "markdown") and res.markdown:
                        if isinstance(res.markdown, dict):
                            # PaddleX 某些版本返回 dict: {'markdown_texts': '...', 'text': '...'}
                            page_md = res.markdown.get('markdown_texts', '') or res.markdown.get('text', '')
                        elif hasattr(res.markdown, 'markdown_texts'):
                            # PaddleX 某些版本返回对象
                            page_md = res.markdown.markdown_texts
                        else:
                            # 或者是直接的字符串
                            page_md = str(res.markdown)
                    elif hasattr(res, "str") and res.str:
                        # 某些特定 pipeline 可能存储在 str 属性
                        page_md = str(res.str)
                except Exception as e:
                    logger.warning(f"⚠️ Page {page_count}: Markdown extraction error: {e}")
                
                # 4. 兜底：尝试读取 save_to_markdown 生成的文件
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
            
            # 保存完整 JSON
            final_json_path = output_path / "result.json"
            combined_data = {
                "total_pages": page_count,
                "pages": json_list
            }
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "result_path": str(output_path),
                "markdown": full_markdown,
                "markdown_file": str(final_md_path),
                "json_file": str(final_json_path)
            }

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """清理显存"""
        try:
            if PADDLE_AVAILABLE and paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
            gc.collect()
            logger.debug("🧹 Memory cleanup completed")
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")

# 全局单例
_engine_instance = None

def get_engine(model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PaddleOCRVLEngine(model_name=model_name)
    return _engine_instance
