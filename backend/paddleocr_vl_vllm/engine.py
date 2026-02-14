"""
PaddleOCR-VL-VLLM 解析引擎
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API
使用最新的 PaddleOCR-VL-VLLM API（自动多语言识别）

参考文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api

重要提示：
- PaddleOCR-VL-VLLM 仅支持 GPU 推理，不支持 CPU 及 Arm 架构
- GPU 要求：Compute Capability ≥ 8.5 (RTX 3090, A10, A100, H100 等)
- 支持本地模型加载（/app/models/paddlex/）或自动下载（持久化到 /root/.paddlex）
"""

import os
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

    GPU 要求：
    - NVIDIA GPU with Compute Capability ≥ 8.5
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
                    logger.info("   Official recommendation: CC ≥ 8.5 for best performance")
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
                logger.info("   ✅ Document Orientation Classification: Enabled")
                logger.info("   ✅ Document Unwarping (Text Correction): Enabled")
                logger.info("   ✅ Layout Detection & Sorting: Enabled")
                logger.info("   ✅ Auto Multi-Language Recognition: Enabled (109+ languages)")

                # =========================================================================
                # 智能路径解析逻辑 (适配 Docker 持久化挂载)
                # =========================================================================
                # 1. 优先检查 Docker 挂载的 PADDLEX_HOME 环境变量
                pdx_home = os.environ.get("PADDLEX_HOME")
                if pdx_home:
                    logger.info(f"💾 Using PADDLEX_HOME from env: {pdx_home}")
                
                # 2. 定义手动模型目录
                base_model_dir = Path("/app/models/paddlex")
                local_model_path = base_model_dir / self.model_name
                
                # PaddleOCRVL 目前版本似乎不直接接受 model_dir 参数作为本地路径
                # 它依赖环境变量 PADDLEX_HOME 去查找或下载模型
                # 但我们还是要检查一下本地是否有模型，以便输出日志
                if local_model_path.exists() and local_model_path.is_dir() and any(local_model_path.iterdir()):
                    logger.info(f"📂 Found local model cache: {local_model_path}")
                else:
                    logger.info(f"🌐 Local model not found at {local_model_path}")
                    logger.info(f"   Will use auto-download to: {pdx_home if pdx_home else 'Default Cache'}")

                # 初始化 PaddleOCRVL
                # 注意：PaddleOCRVL 内部会使用 PADDLEX_HOME 环境变量来决定下载/加载位置
                self._pipeline = PaddleOCRVL(
                    use_doc_orientation_classify=True,  # 文档方向分类
                    use_doc_unwarping=True,             # 文本图像矫正
                    use_layout_detection=True,          # 版面区域检测
                    vl_rec_backend="vllm-server",       # 使用 VLLM 后端
                    vl_rec_server_url=self.vllm_api_base, # VLLM 服务器地址
                )
                
                logger.info("=" * 60)
                logger.info("✅ PaddleOCR-VL-VLLM Pipeline loaded successfully!")
                logger.info(f"   Device: GPU {self.gpu_id}")
                logger.info("   Features: Orientation correction, Text unwarping, Layout detection")
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
            **kwargs: 其他参数

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

        # 执行推理
        try:
            logger.info("🚀 开始使用 PaddleOCR-VL-VLLM 识别...")
            logger.info(f"   输入文件: {file_path}")
            logger.info("   自动语言检测: 支持 109+ 语言")

            # PaddleOCR-VL-VLLM 的 predict 方法
            result = pipeline.predict(str(file_path))

            logger.info("✅ PaddleOCR-VL-VLLM completed")
            logger.info(f"   识别了 {len(result)} 页/张")

            # 处理结果
            markdown_list = []
            json_list = []

            for idx, res in enumerate(result, 1):
                logger.info(f"📝 处理结果 {idx}/{len(result)}")

                try:
                    page_output_dir = output_path / f"page_{idx}"
                    page_output_dir.mkdir(parents=True, exist_ok=True)

                    if hasattr(res, "save_to_json"):
                        res.save_to_json(save_path=str(page_output_dir))
                    
                    if hasattr(res, "save_to_markdown"):
                        res.save_to_markdown(save_path=str(page_output_dir))

                    if hasattr(res, "markdown"):
                        md_info = res.markdown
                        markdown_list.append(md_info)
                    
                    if hasattr(res, "json"):
                        json_list.append(res.json)

                except Exception as e:
                    logger.warning(f"   处理出错: {e}")

            # 合并 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
                    logger.info("   使用官方 concatenate_markdown_pages() 方法合并")
                except Exception:
                     markdown_text = "\n\n---\n\n".join([str(m) for m in markdown_list])
            else:
                markdown_text = "\n\n---\n\n".join([str(m) for m in markdown_list])

            # 保存结果
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            logger.info(f"📄 Markdown 已保存: {markdown_file}")

            json_file = output_path / "result.json"
            if json_list:
                import json as json_lib
                combined_json = {"pages": json_list, "total_pages": len(result)}
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
            import traceback
            logger.debug(traceback.format_exc())
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
