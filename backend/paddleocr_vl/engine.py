"""
PaddleOCR-VL 解析引擎
单例模式，每个进程只加载一次模型
使用最新的 PaddleOCR-VL API（自动多语言识别）

参考文档：http://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html

重要提示：
- PaddleOCR-VL 仅支持 GPU 推理，不支持 CPU 及 Arm 架构
- GPU 要求：Compute Capability ≥ 8.5 (RTX 3090, A10, A100, H100 等)
- 支持本地模型加载（/app/models/paddlex/）或自动下载
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
import time
from loguru import logger


class PaddleOCRVLEngine:
    """
    PaddleOCR-VL 解析引擎（新版本）

    特性：
    - 单例模式（每个进程只加载一次模型）
    - 自动多语言识别（无需指定语言，支持 109+ 语言）
    - 线程安全
    - 仅支持 GPU 推理（不支持 CPU）
    - 原生支持 PDF 多页文档
    - 结构化输出（Markdown/JSON）
    - 支持加载本地模型缓存，避免重复下载

    GPU 要求：
    - NVIDIA GPU with Compute Capability ≥ 8.5
    - 推荐：RTX 3090, RTX 4090, A10, A100, H100
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
        """
        初始化引擎（只执行一次）

        Args:
            device: 设备 (cuda:0, cuda:1 等，PaddleOCR 仅支持 GPU)
            model_name: 模型名称或路径 (默认: PaddleOCR-VL-1.5-0.9B)
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.model_name = model_name

            # 从 device 字符串中提取 GPU ID (例如 "cuda:0" -> 0)
            if "cuda:" in device:
                self.gpu_id = int(device.split(":")[-1])
            else:
                self.gpu_id = 0
                logger.warning(f"⚠️  Invalid device format: {device}, using GPU 0")

            # 检查 GPU 可用性（PaddleOCR-VL 仅支持 GPU）
            self._check_gpu_availability()

            self._initialized = True

            logger.info("🔧 PaddleOCR-VL Engine initialized")
            logger.info(f"   Device: {self.device} (GPU ID: {self.gpu_id})")
            logger.info(f"   Target Model: {self.model_name}")

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
        """延迟加载 PaddleOCR-VL 管道"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("=" * 60)
            logger.info(f"📥 Loading PaddleOCR-VL Pipeline ({self.model_name})...")
            logger.info("=" * 60)

            try:
                import paddle
                from paddlex import create_pipeline

                # 设置 PaddlePaddle 使用指定的 GPU
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")
                    logger.info(f"🎯 PaddlePaddle device set to: gpu:{self.gpu_id}")
                else:
                    logger.warning("⚠️  CUDA not available, PaddleOCR-VL may not work")

                # =========================================================================
                # 智能路径解析逻辑
                # =========================================================================
                # 1. 定义本地模型根目录 (指向 paddlex 子目录)
                base_model_dir = Path("/app/models/paddlex")
                
                # 2. 尝试拼接本地路径
                # PaddleX 官方模型名称通常包含版本号
                local_model_path = base_model_dir / self.model_name
                
                pipeline_source = self.model_name # 默认使用模型名称（触发在线查找/缓存查找）

                if local_model_path.exists() and local_model_path.is_dir():
                    logger.info(f"📂 Found local model cache: {local_model_path}")
                    # 强制使用本地绝对路径，防止 PaddleX 重新去 ~/.paddlex 下载
                    pipeline_source = str(local_model_path)
                    
                    # 设置 PaddleX 缓存目录到 paddlex 下，保持整洁
                    # 这样即使下载辅助模型，也会存放在我们挂载的目录中
                    os.environ["PADDLEX_HOME"] = "/app/models/paddlex/.paddlex_cache"
                else:
                    logger.warning(f"⚠️  Local model not found at: {local_model_path}")
                    logger.info(f"   Will attempt to load '{self.model_name}' from official source/cache...")

                # 初始化管道
                start_time = time.time()
                
                # 使用 PaddleX 的 create_pipeline API
                self._pipeline = create_pipeline(
                    pipeline=pipeline_source,
                    device=f"gpu:{self.gpu_id}" if paddle.is_compiled_with_cuda() else "cpu",
                    # 可以在这里传递其他参数，例如 use_hp_ip=True 等
                )

                logger.info("=" * 60)
                logger.info(f"✅ PaddleOCR-VL Pipeline loaded in {time.time() - start_time:.2f}s!")
                logger.info(f"   Source: {pipeline_source}")
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
                logger.error("   1. 确保模型文件存在于 /app/models/paddlex/")
                logger.error("   2. 检查显存是否充足")
                logger.error("   3. 检查 CUDA 版本兼容性")
                logger.error("=" * 80)

                import traceback
                logger.debug("完整堆栈跟踪:")
                logger.debug(traceback.format_exc())

                raise

    def cleanup(self):
        """
        清理推理产生的显存（不卸载模型）
        """
        try:
            import paddle
            import gc

            # 清理 PaddlePaddle 显存
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.debug("🧹 PaddleOCR-VL: CUDA cache cleared")

            # 清理 Python 对象
            gc.collect()

            logger.debug("🧹 PaddleOCR-VL: Memory cleanup completed")
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

        logger.info(f"🤖 PaddleOCR-VL parsing: {file_path.name}")
        
        # 加载管道
        pipeline = self._load_pipeline()

        # 执行推理
        try:
            logger.info("🚀 开始使用 PaddleOCR-VL 识别...")
            
            # PaddleX v3 predict 方法参数
            # use_doc_orientation_classify: 启用文档方向分类
            # use_doc_unwarping: 启用文档矫正
            # use_layout_parsing: 启用版面分析
            result = pipeline.predict(
                str(file_path),
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_layout_parsing=True
            )
            
            # 结果可能是一个生成器或列表
            results = list(result)

            logger.info("✅ PaddleOCR-VL completed")
            logger.info(f"   识别了 {len(results)} 页/张")

            markdown_list = []
            json_list = []

            for idx, res in enumerate(results, 1):
                logger.info(f"📝 处理结果 {idx}/{len(results)}")

                try:
                    # 为每页创建子目录
                    page_output_dir = output_path / f"page_{idx}"
                    page_output_dir.mkdir(parents=True, exist_ok=True)

                    # 保存可视化结果和JSON
                    if hasattr(res, "save_to_img"):
                        res.save_to_img(str(page_output_dir))
                    if hasattr(res, "save_to_json"):
                        res.save_to_json(str(page_output_dir))
                    
                    # 尝试保存 Markdown (如果支持)
                    if hasattr(res, "save_to_markdown"):
                        res.save_to_markdown(str(page_output_dir))

                    # 收集 Markdown 内容
                    # 注意：PaddleX 不同版本的属性名可能不同，尝试做兼容
                    if hasattr(res, "markdown"):
                        # 如果 res.markdown 是对象，尝试转字符串或取 text
                        md_content = res.markdown
                        markdown_list.append(md_content)
                    elif hasattr(res, "str"):
                         # 某些任务可能用 str 属性返回文本
                         markdown_list.append(res.str)

                    # 收集 JSON 数据 (如果有)
                    # 有些结果对象可能没有直接的 json 属性，而是通过 save_to_json 生成
                    # 这里尝试从已保存的文件读取，或者如果对象有 dict/json 方法
                    # 简单起见，如果 res 本身是可序列化的，也可以直接用
                    pass 

                except Exception as e:
                    logger.warning(f"   页处理出错: {e}")
            
            # 合并 Markdown
            markdown_text = ""
            
            # 优先使用 pipeline 自带的合并方法（如果存在）
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
                    logger.info("   使用官方 concatenate_markdown_pages() 方法合并")
                except Exception as e:
                    logger.warning(f"   合并失败，降级为手动合并: {e}")
                    markdown_text = "\n\n---\n\n".join([str(m) for m in markdown_list])
            elif markdown_list:
                # 手动合并
                markdown_text = "\n\n---\n\n".join([str(m) for m in markdown_list])
            
            # 如果没有直接获得 markdown，尝试读取生成的 .md 文件
            if not markdown_text:
                logger.info("   尝试从输出目录读取 Markdown 文件...")
                for md_file in output_path.rglob("*.md"):
                    if md_file.name != "result.md": # 排除自己
                        text = md_file.read_text(encoding="utf-8")
                        markdown_text += text + "\n\n---\n\n"

            # 保存最终结果
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            logger.info(f"📄 Markdown 已保存: {markdown_file}")

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                # "json_file": ... (如果生成了合并的JSON)
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


def get_engine(model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLEngine:
    """
    获取全局引擎实例
    注意：单例模式下，第一次调用时的 model_name 会决定后续一直使用的模型
    """
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLEngine(model_name=model_name)
    return _engine
