"""
PaddleOCR-VL-VLLM 解析引擎
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API
使用最新的 PaddleOCR-VL-VLLM API（自动多语言识别）

参考文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api

重要提示：
- PaddleOCR-VL-VLLM 仅支持 GPU 推理，不支持 CPU 及 Arm 架构
- GPU 要求：Compute Capability >= 8.5 (RTX 3090, A10, A100, H100 等)
- 支持本地模型加载（/root/.paddlex/official_models/）或自动下载
"""

import os
import gc
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
import time
from loguru import logger

class PaddleOCRVLVLLMEngine:
    """
    PaddleOCR-VL-VLLM 解析引擎（新版本）

    特性：
    - 单例模式（每个进程只加载一次模型）
    - 自动多语言识别（无需指定语言，支持 109+ 语言）
    - 线程安全
    - 仅支持 GPU 推理（不支持 CPU）
    - 原生支持 PDF 多页文档解析
    - 结构化输出（Markdown/JSON）
    - 模型自动下载和缓存（支持持久化挂载）
    - 内存优化：使用生成器流式处理长文档，防止 OOM
    - 参数支持：支持 PaddleOCR-VL-1.5 的全量高级参数配置

    GPU 要求：
    - NVIDIA GPU with Compute Capability >= 8.5
    - 推荐：RTX 3090, RTX 4090, A10, A100, H100
    """

    _instance: Optional["PaddleOCRVLVLLMEngine"] = None
    _lock = Lock()
    _pipeline = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0", vllm_api_base: str = "http://localhost:17300/v1", model_name: str = "PaddleOCR-VL-1.5-0.9B"):
        """
        初始化引擎（只执行一次）

        Args:
            device: 设备 (cuda:0, cuda:1 等，PaddleOCR 仅支持 GPU)
            vllm_api_base: VLLM API 基础 URL
            model_name: 模型名称 (默认: PaddleOCR-VL-1.5-0.9B)
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.vllm_api_base = vllm_api_base
            self.model_name = model_name

            # 从 device 字符串中提取 GPU ID (例如 "cuda:0" -> 0)
            if "cuda:" in device:
                self.gpu_id = int(device.split(":")[-1])
            else:
                self.gpu_id = 0
                logger.warning(f"⚠️  Invalid device format: {device}, using GPU 0")

            # 检查 GPU 可用性
            self._check_gpu_availability()

            self._initialized = True

            logger.info("🔧 PaddleOCR-VL-VLLM Engine initialized")
            logger.info(f"   Device: {self.device} (GPU ID: {self.gpu_id})")
            logger.info(f"   VLLM API Base: {self.vllm_api_base}")
            logger.info(f"   Model: {self.model_name}")

    def _check_gpu_availability(self):
        """
        检查 GPU 信息并输出日志
        PaddleOCR-VL 仅支持 GPU 推理，但不阻止低版本 GPU 运行
        """
        try:
            import paddle

            # 检查是否编译了 CUDA 支持
            if not paddle.is_compiled_with_cuda():
                logger.warning("⚠️  PaddlePaddle is not compiled with CUDA")
                logger.warning("   PaddleOCR-VL requires GPU support")
                logger.warning("   Install: pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
                return

            # 检查是否有可用的 GPU
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count == 0:
                logger.warning("⚠️  No CUDA devices found")
                logger.warning("   PaddleOCR-VL requires GPU for inference")
                return

            # 获取 GPU 信息
            try:
                gpu_name = paddle.device.cuda.get_device_name(0)
                compute_capability = paddle.device.cuda.get_device_capability(0)

                logger.info(f"✅ GPU detected: {gpu_name}")
                logger.info(f"   Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
                logger.info(f"   GPU Count: {gpu_count}")

                # 仅输出建议，不阻止运行
                cc_major = compute_capability[0]
                cc_minor = compute_capability[1]
                if cc_major < 8 or (cc_major == 8 and cc_minor < 5):
                    logger.info("ℹ️  GPU Compute Capability < 8.5")
                    logger.info("   Official recommendation: CC >= 8.5 for best performance")
                    logger.info("   Your GPU may still work, but performance might vary")
            except Exception as e:
                logger.debug(f"Could not get detailed GPU info: {e}")

        except ImportError:
            logger.warning("⚠️  PaddlePaddle not installed")
        except Exception as e:
            logger.debug(f"GPU check warning: {e}")

    def _load_pipeline(self):
        """延迟加载 PaddleOCR-VL-VLLM 管道"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("=" * 60)
            logger.info("📥 Loading PaddleOCR-VL-VLLM Pipeline into memory...")
            logger.info("=" * 60)

            try:
                import paddle
                from paddleocr import PaddleOCRVL

                # 设置 PaddlePaddle 使用指定的 GPU
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")
                    logger.info(f"🎯 PaddlePaddle device set to: gpu:{self.gpu_id}")
                else:
                    logger.warning("⚠️  CUDA not available, PaddleOCR-VL may not work")

                if self.vllm_api_base is None:
                    raise ValueError("vllm_api_base cannot be None for VLLM engine")

                logger.info("🤖 Initializing PaddleOCR-VL-VLLM with enhanced features...")
                
                # =========================================================================
                # 智能路径解析逻辑 (适配 Docker 持久化挂载)
                # =========================================================================
                # 1. 获取 PADDLEX_HOME 环境变量，默认指向 /root/.paddlex
                pdx_home = os.environ.get("PADDLEX_HOME", "/root/.paddlex")
                logger.info(f"💾 Using PADDLEX_HOME: {pdx_home}")
                
                # 2. 修正为真实的 PaddleX 官方模型缓存目录
                base_model_dir = Path(pdx_home) / "official_models"
                local_model_path = base_model_dir / self.model_name
                
                # 探测本地是否有模型，以便输出准确的日志
                if local_model_path.exists() and local_model_path.is_dir() and any(local_model_path.iterdir()):
                    logger.info(f"📂 Found local model cache: {local_model_path}")
                else:
                    logger.warning(f"🌐 Local model not found at {local_model_path}")
                    logger.info("   Will attempt auto-download...")

                # 初始化 PaddleOCRVL
                # (预测时的高级参数将通过 predict(**kwargs) 传递)
                self._pipeline = PaddleOCRVL(
                    vl_rec_backend="vllm-server",       # 使用 VLLM 后端
                    vl_rec_server_url=self.vllm_api_base, # VLLM 服务器地址
                )
                
                logger.info("=" * 60)
                logger.info("✅ PaddleOCR-VL-VLLM Pipeline loaded successfully!")
                logger.info(f"   Device: GPU {self.gpu_id}")
                logger.info("=" * 60)

                return self._pipeline

            except Exception as e:
                logger.error("=" * 80)
                logger.error("❌ 管道加载失败:")
                logger.error(f"   错误类型: {type(e).__name__}")
                logger.error(f"   错误信息: {e}")
                logger.error("")
                logger.error("💡 排查建议:")
                logger.error("   1. 检查 vLLM 服务是否启动 (http://vllm-paddleocr:30023)")
                logger.error("   2. 检查网络连接（首次运行需要下载版面分析模型）")
                logger.error("   3. 检查显存是否充足")
                logger.error("=" * 80)

                import traceback
                logger.debug("完整堆栈跟踪:")
                logger.debug(traceback.format_exc())

                raise

    def warmup(self):
        """
        手动触发模型加载（预热）
        """
        if self._pipeline is None:
            logger.info("🔥 Warming up PaddleOCR-VL-VLLM engine...")
            try:
                self._load_pipeline()
                logger.info("🔥 Warmup completed! Engine is ready.")
            except Exception as e:
                logger.error(f"🔥 Warmup failed: {e}")

    def cleanup(self):
        """
        清理推理产生的显存（不卸载模型）
        """
        try:
            import paddle
            import gc

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.debug("🧹 PaddleOCR-VL-VLLM: CUDA cache cleared")

            gc.collect()
            logger.debug("🧹 PaddleOCR-VL-VLLM: Memory cleanup completed")
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析文档或图片

        Args:
            file_path: 输入文件路径
            output_path: 输出目录
            **kwargs: 其他高级控制参数

        Returns:
            解析结果（同时保存 Markdown 和 JSON 两种格式）
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🤖 PaddleOCR-VL-VLLM parsing: {file_path.name}")
        logger.info("   Auto language detection enabled")

        # 加载管道
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
            "promptLabel": "prompt_label", 
            "temperature": "temperature",
            "topP": "top_p",
            "repetitionPenalty": "repetition_penalty"
        }

        # 规范化参数并过滤天枢其他无关参数
        predict_params = {}
        for k, v in kwargs.items():
            if k in param_mapping:
                predict_params[param_mapping[k]] = v

        try:
            # 动态检查预处理模块是否支持
            has_preprocessor = hasattr(pipeline, "doc_preprocessor_pipeline") and pipeline.doc_preprocessor_pipeline is not None
            req_orientation = predict_params.get("use_doc_orientation_classify", False)
            req_unwarping = predict_params.get("use_doc_unwarping", False)

            if (req_orientation or req_unwarping) and not has_preprocessor:
                logger.warning("⚠️ 请求了文档矫正/分类，但模型缺少预处理模块。已自动禁用以防止崩溃。")
                predict_params["use_doc_orientation_classify"] = False
                predict_params["use_doc_unwarping"] = False

            # 设置输入和基本默认值
            predict_params["input"] = str(file_path)
            if "use_layout_parsing" not in predict_params: predict_params["use_layout_parsing"] = True
            if "use_doc_orientation_classify" not in predict_params: predict_params["use_doc_orientation_classify"] = False
            if "use_doc_unwarping" not in predict_params: predict_params["use_doc_unwarping"] = False

            log_params = {k: v for k, v in predict_params.items() if k != "input"}
            logger.info(f"🚀 开始使用 PaddleOCR-VL-VLLM 识别 (参数: {json.dumps(log_params, default=str, ensure_ascii=False)})")

            # 执行推理 (使用流式生成器防止长文档 OOM)
            output_generator = pipeline.predict(**predict_params)

            markdown_pages = []
            markdown_list_obj = [] # 用于保存原始 markdown 对象以便进行官方合并
            json_list = []
            page_count = 0

            for res in output_generator:
                page_count += 1
                logger.info(f"📝 处理结果 第 {page_count} 页")
                page_output_dir = output_path / f"page_{page_count}"
                page_output_dir.mkdir(parents=True, exist_ok=True)

                # 保存文件
                if hasattr(res, "save_to_img"): res.save_to_img(str(page_output_dir))
                if hasattr(res, "save_to_json"): res.save_to_json(str(page_output_dir))

                # 收集 JSON 对象
                if hasattr(res, "json"):
                    json_list.append(res.json)

                # 收集 Markdown 对象和字符串
                if hasattr(res, "markdown") and res.markdown:
                    markdown_list_obj.append(res.markdown)
                
                # 健壮提取当前页 Markdown
                page_md = ""
                if hasattr(res, "markdown") and res.markdown:
                    if isinstance(res.markdown, dict):
                        page_md = res.markdown.get('markdown_texts', '') or res.markdown.get('text', '')
                    elif hasattr(res.markdown, 'markdown_texts'):
                        page_md = res.markdown.markdown_texts
                    elif isinstance(res.markdown, str):
                        page_md = res.markdown
                    else:
                        page_md = str(res.markdown)
                elif hasattr(res, "str") and res.str:
                    page_md = str(res.str)

                # 兜底文件读取
                if not page_md and hasattr(res, "save_to_markdown"):
                    try:
                        res.save_to_markdown(str(page_output_dir))
                        saved_mds = list(page_output_dir.glob("*.md"))
                        if saved_mds:
                            page_md = saved_mds[0].read_text(encoding="utf-8")
                    except Exception:
                        pass

                if page_md:
                    markdown_pages.append(page_md)
                else:
                    logger.warning(f"⚠️ Page {page_count}: No markdown content extracted.")

            logger.info(f"✅ PaddleOCR-VL-VLLM completed, Processed {page_count} pages")

            # 合并 Markdown
            markdown_text = ""
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list_obj:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list_obj)
                    logger.info("   使用官方 concatenate_markdown_pages() 方法合并")
                except Exception as e:
                    logger.warning(f"官方合并方法失败: {e}, 自动回退到常规拼接")
                    markdown_text = "\n\n---\n\n".join(markdown_pages)
            else:
                markdown_text = "\n\n---\n\n".join(markdown_pages)

            # 保存最终结果
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            logger.info(f"📄 Markdown 已保存: {markdown_file}")

            json_file = output_path / "result.json"
            if json_list:
                import json as json_lib
                combined_json = {"pages": json_list, "total_pages": page_count}
                with open(json_file, "w", encoding="utf-8") as f:
                    json_lib.dump(combined_json, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 JSON 已保存: {json_file}")

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file),
            }

        except Exception as e:
            logger.error(f"❌ OCR 解析失败: {e}")
            logger.error(traceback.format_exc())
            raise

        finally:
            self.cleanup()

# 全局单例
_engine = None

def get_engine(vllm_api_base: str = "http://localhost:17300/v1", model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
