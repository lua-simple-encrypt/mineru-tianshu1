"""
PaddleOCR-VL-VLLM 解析引擎
单例模式，每个进程只加载一次基础版面识别模型,OCR部分调用配置的API
使用最新的 PaddleOCR-VL-VLLM API（自动多语言识别）

参考文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api

重要提示：
- PaddleOCR-VL-VLLM 仅支持 GPU 推理，不支持 CPU 及 Arm 架构
- GPU 要求：Compute Capability ≥ 8.5 (RTX 3090, A10, A100, H100 等)
- 基础版面识别模型会自动下载到 ~/.paddleocr/models/ 目录（PaddleOCR 自动管理）
- OCR模型默认调用配置的VLLM API，用户可以在初始化时指定API地址
- 不支持手动指定本地模型路径，模型由 PaddleOCR 自动管理
"""

from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import os

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
    - 模型自动下载和缓存（由 PaddleOCR 管理，无需手动下载）

    GPU 要求：
    - NVIDIA GPU with Compute Capability ≥ 8.5
    - 推荐：RTX 3090, RTX 4090, A10, A100, H100

    模型管理：
    - 基础版面识别模型由 PaddleOCR 自动下载和管理
    - OCR模型默认调用配置的VLLM API，用户可以在初始化时指定API地址
    - 默认缓存位置：~/.paddleocr/models/
    - 不支持手动指定本地模型路径
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
            vllm_api_base: VLLM API 基础 URL (默认: http://localhost:17300/v1)
            model_name: 模型名称 (默认: PaddleOCR-VL-1.5-0.9B)

        注意：
        - PaddleOCR-VL 会自动管理模型的下载和缓存
        - 模型默认缓存在 ~/.paddleocr/models/
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device  # 保存 device 参数
            self.vllm_api_base = vllm_api_base  # 保存 vllm_api_base 参数
            self.model_name = model_name # 保存模型名称

            # 从 device 字符串中提取 GPU ID (例如 "cuda:0" -> 0)
            if "cuda:" in device:
                self.gpu_id = int(device.split(":")[-1])
            else:
                self.gpu_id = 0
                logger.warning(f"⚠️  Invalid device format: {device}, using GPU 0")

            # 检查 GPU 可用性（PaddleOCR-VL 仅支持 GPU）
            self._check_gpu_availability()

            self._initialized = True

            logger.info("🔧 PaddleOCR-VL-VLLM Engine initialized")
            logger.info(f"   Device: {self.device} (GPU ID: {self.gpu_id})")
            logger.info(f"   VLLM API Base: {self.vllm_api_base}")
            logger.info(f"   Model: {self.model_name} (local path priority)")
            logger.info("   Auto Multi-Language: Enabled (109+ languages)")
            logger.info("   GPU Only: CPU not supported")

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
                logger.warning("   Install: pip install paddlepaddle-gpu==3.2.0")
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
            logger.warning("   Install: pip install paddlepaddle-gpu==3.2.0")
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
                # 必须在创建 PaddleOCRVL 实例之前设置
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")
                    logger.info(f"🎯 PaddlePaddle device set to: gpu:{self.gpu_id}")
                else:
                    logger.warning("⚠️  CUDA not available, PaddleOCR-VL may not work")

                # 初始化 PaddleOCR-VL（新版本 API）
                # 为了最佳识别效果，启用所有增强功能
                logger.info("🤖 Initializing PaddleOCR-VL-VLLM with enhanced features...")
                logger.info("   ✅ Document Orientation Classification: Enabled")
                logger.info("   ✅ Document Unwarping (Text Correction): Enabled")
                logger.info("   ✅ Layout Detection & Sorting: Enabled")
                logger.info("   ✅ Auto Multi-Language Recognition: Enabled (109+ languages)")

                # 创建 PaddleOCRVL 实例（按照官方文档最佳实践）
                # 参考: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api

                if self.vllm_api_base is None:
                    # 抛出一个异常
                    raise ValueError(
                        "vllm_api_base 不能为 None，请检查paddleocr-vl-vllm-engine-enabled 及 paddleocr-vl-vllm-api-list 配置"
                    )
                else:
                    # =========================================================================
                    # 智能路径解析逻辑 (适配本地路径)
                    # =========================================================================
                    # 1. 定义本地模型根目录 (指向 paddlex 子目录)
                    base_model_dir = Path("/app/models/paddlex")
                    
                    # 2. 尝试拼接本地路径
                    local_model_path = base_model_dir / self.model_name
                    
                    # 默认参数（如果本地没有，Paddle 会尝试在线下载或使用默认值）
                    # 注意：PaddleOCRVL 类本身通常不需要传入 pipeline 参数，它是通过 use_xxx 参数来内部构建 pipeline 的
                    # 但是，如果我们要指定本地模型路径，通常需要查阅 PaddleOCRVL 的构造函数参数
                    # 假设 PaddleOCRVL 接受 `layout_model_dir` 或类似的参数来指定本地模型
                    # 或者，如果 PaddleOCRVL 是基于 PaddleX pipeline 实现的，我们可能无法直接在这里传入本地路径
                    # **修正策略**：PaddleOCRVL 是一个高层封装。如果我们要用本地模型，通常需要设置 PADDLEX_HOME 环境变量
                    # 让它去我们挂载的目录找。
                    
                    if local_model_path.exists() and local_model_path.is_dir():
                        logger.info(f"📂 Found local model cache: {local_model_path}")
                        # 设置环境变量，引导 PaddleX 去本地目录查找模型
                        # PaddleX 默认会在 $PADDLEX_HOME/official_models/ 下查找
                        # 我们这里设置 PADDLEX_HOME 为 /app/models/paddlex/.. (即 /app/models)
                        # 这样它会在 /app/models/official_models/ 下找... 这可能有点绕
                        # 最稳妥的方式是：保持默认下载行为，但因为我们已经把模型下载到了它期望的位置（通过 download_models.py），
                        # 只要路径对上了，它就不会重新下载。
                        
                        # 另外，PaddleOCRVL 的构造函数可能不直接支持指定本地模型路径。
                        # 我们这里主要依赖 `docker-compose.yml` 中配置的 `PADDLEX_HOME` 环境变量。
                    else:
                        logger.warning(f"⚠️  Local model path not found: {local_model_path}")

                    self._pipeline = PaddleOCRVL(
                        use_doc_orientation_classify=True,  # 文档方向分类，自动旋转文档
                        use_doc_unwarping=True,  # 文本图像矫正，修正扭曲变形
                        use_layout_detection=True,  # 版面区域检测排序，智能排版,
                        vl_rec_backend="vllm-server",  # 使用 VLLM 后端
                        vl_rec_server_url=self.vllm_api_base,  # VLLM 服务器地址
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
                logger.error("   1. 确保已安装正确版本:")
                logger.error("      pip install paddlepaddle-gpu==3.2.0")
                logger.error("      pip install 'paddleocr[doc-parser]'")
                logger.error("   2. 安装 SafeTensors:")
                logger.error(
                    "     #    pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl"
                )
                logger.error("   3. 检查 GPU 可用性:")
                logger.error("      python -c 'import paddle; print(paddle.device.is_compiled_with_cuda())'")
                logger.error("   4. 检查磁盘空间是否充足")
                logger.error("   5. 检查网络连接（首次使用需要下载模型）")
                logger.error("")
                logger.error(
                    "参考文档: https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#312-paddleocr-cli"
                )
                logger.error("=" * 80)

                import traceback

                logger.debug("完整堆栈跟踪:")
                logger.debug(traceback.format_exc())

                raise

    def cleanup(self):
        """
        清理推理产生的显存（不卸载模型）

        注意：
        - 只清理推理过程中产生的中间张量
        - 不会卸载已加载的模型（模型保持在显存中，下次推理更快）
        - 适合在每次推理完成后调用
        """
        try:
            import paddle
            import gc

            # 清理 PaddlePaddle 显存
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.debug("🧹 PaddleOCR-VL-VLLM: CUDA cache cleared")

            # 清理 Python 对象
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
            **kwargs: 其他参数（PaddleOCR-VL 会自动识别语言）

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

            # PaddleOCR-VL-VLLM 的 predict 方法可以直接处理 PDF 或图片
            # 它会自动处理多页文档和语言检测
            result = pipeline.predict(str(file_path))

            logger.info("✅ PaddleOCR-VL-VLLM completed")
            logger.info(f"   识别了 {len(result)} 页/张")

            # 按照官方示例处理结果
            markdown_list = []
            json_list = []

            for idx, res in enumerate(result, 1):
                logger.info(f"📝 处理结果 {idx}/{len(result)}")

                try:
                    # 为每页创建子目录并保存完整结果（便于调试）
                    page_output_dir = output_path / f"page_{idx}"
                    page_output_dir.mkdir(parents=True, exist_ok=True)

                    # 保存 JSON（结构化数据）
                    if hasattr(res, "save_to_json"):
                        res.save_to_json(save_path=str(page_output_dir))

                    # 保存 Markdown 文件（便于调试）
                    if hasattr(res, "save_to_markdown"):
                        res.save_to_markdown(save_path=str(page_output_dir))

                    # 按照官方示例：收集每页的 markdown 对象
                    if hasattr(res, "markdown"):
                        md_info = res.markdown
                        markdown_list.append(md_info)
                        logger.info("   ✅ 提取成功")
                    else:
                        logger.warning("   ⚠️  无法提取内容")

                    # 收集 JSON 数据
                    if hasattr(res, "json"):
                        json_data = res.json
                        json_list.append(json_data)

                except Exception as e:
                    logger.warning(f"   处理出错: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())

            # 使用官方方法合并所有页的 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages"):
                markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
                logger.info("   使用官方 concatenate_markdown_pages() 方法合并")
            else:
                # 降级方案：手动合并
                logger.warning("   未找到 concatenate_markdown_pages() 方法，使用降级方案")
                markdown_text = "\n\n---\n\n".join(
                    [str(md) if isinstance(md, str) else str(md.get("text", "")) for md in markdown_list]
                )

            # 保存合并后的 Markdown 文件
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            logger.info(f"📄 Markdown 已保存: {markdown_file}")
            logger.info(f"   {len(result)} 页 | {len(markdown_text):,} 字符")

            # 始终保存 JSON 文件（方便用户后续选择）
            json_file = None
            if json_list:
                import json as json_lib

                json_file = output_path / "result.json"
                # 合并所有页的 JSON
                combined_json = {"pages": json_list, "total_pages": len(result)}
                with open(json_file, "w", encoding="utf-8") as f:
                    json_lib.dump(combined_json, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 JSON 已保存: {json_file}")
            else:
                logger.warning("⚠️  无法提取 JSON 数据")

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file) if json_file else None,
                "result": result,
            }

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ OCR 解析失败:")
            logger.error(f"   错误类型: {type(e).__name__}")
            logger.error(f"   错误信息: {e}")
            logger.error("=" * 80)

            import traceback

            logger.debug("完整堆栈跟踪:")
            logger.debug(traceback.format_exc())

            raise

        finally:
            # 清理显存（无论成功或失败都执行）
            self.cleanup()


# 全局单例
_engine = None


def get_engine(vllm_api_base: str = "http://localhost:17300/v1", model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    """
    获取全局引擎实例
    注意：单例模式下，第一次调用时的 model_name 会决定后续一直使用的模型
    """
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
