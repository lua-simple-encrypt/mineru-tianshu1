"""
MinerU Tianshu - LitServe Worker
天枢 LitServe Worker

企业级 AI 数据预处理平台 - GPU Worker
支持文档、图片、音频、视频等多模态数据处理
使用 LitServe 实现 GPU 资源的自动负载均衡
Worker 主动循环拉取任务并处理
"""

import os
import json
import sys
import time
import threading
import signal
import atexit
from pathlib import Path
from typing import Optional
import multiprocessing

# Fix litserve MCP compatibility with mcp>=1.1.0
# Completely disable LitServe's internal MCP to avoid conflicts with our standalone MCP Server
import litserve as ls
from litserve.connector import check_cuda_with_nvidia_smi
from utils import parse_list_arg

try:
    # Patch LitServe's MCP module to disable it completely
    import litserve.mcp as ls_mcp
    import sys
    from contextlib import asynccontextmanager

    # Inject MCPServer (mcp.server.lowlevel.Server) as dummy
    if not hasattr(ls_mcp, "MCPServer"):

        class DummyMCPServer:
            def __init__(self, *args, **kwargs):
                pass

        ls_mcp.MCPServer = DummyMCPServer
        if "litserve.mcp" in sys.modules:
            sys.modules["litserve.mcp"].MCPServer = DummyMCPServer

    # Inject StreamableHTTPSessionManager as dummy
    if not hasattr(ls_mcp, "StreamableHTTPSessionManager"):

        class DummyStreamableHTTPSessionManager:
            def __init__(self, *args, **kwargs):
                pass

        ls_mcp.StreamableHTTPSessionManager = DummyStreamableHTTPSessionManager
        if "litserve.mcp" in sys.modules:
            sys.modules["litserve.mcp"].StreamableHTTPSessionManager = DummyStreamableHTTPSessionManager

    # Replace _LitMCPServerConnector with a complete dummy implementation
    class DummyMCPConnector:
        """完全禁用 LitServe 内置 MCP 的 Dummy 实现"""

        def __init__(self, *args, **kwargs):
            self.mcp_server = None
            self.session_manager = None
            self.request_handler = None

        @asynccontextmanager
        async def lifespan(self, app):
            """空的 lifespan context manager，不做任何事情"""
            yield  # 什么都不做，直接让服务器启动

        def connect_mcp_server(self, *args, **kwargs):
            """空的 connect_mcp_server 方法，不做任何事情"""
            pass  # 什么都不做，跳过 MCP 初始化

    # 替换 _LitMCPServerConnector 类
    ls_mcp._LitMCPServerConnector = DummyMCPConnector

    # 同时更新 sys.modules 中的引用
    if "litserve.mcp" in sys.modules:
        sys.modules["litserve.mcp"]._LitMCPServerConnector = DummyMCPConnector

except Exception as e:
    # If patching fails, log warning and continue
    # The server might still work or fail with a clearer error message
    import warnings

    warnings.warn(f"Failed to patch litserve.mcp (MCP will be disabled): {e}")

from loguru import logger

# 添加父目录到路径以导入 MinerU
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from task_db import TaskDB
from output_normalizer import normalize_output

# 延迟导入 MinerU，避免过早初始化 CUDA
# MinerU 会在 setup() 设置 CUDA_VISIBLE_DEVICES 后再导入
# from mineru.cli.common import do_parse
# from mineru.utils.model_utils import get_vram, clean_memory

# 导入 importlib 用于检查模块可用性
import importlib.util

# 尝试导入 markitdown
try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logger.warning("⚠️  markitdown not available, Office format parsing will be disabled")

# 检查 PaddleOCR-VL 是否可用（不要导入，避免初始化 CUDA）
PADDLEOCR_VL_AVAILABLE = importlib.util.find_spec("paddleocr_vl") is not None
if PADDLEOCR_VL_AVAILABLE:
    logger.info("✅ PaddleOCR-VL engine available")
else:
    logger.info("ℹ️  PaddleOCR-VL not available (optional)")

# 检查 PaddleOCR-VL-VLLM 是否可用（不要导入，避免初始化 CUDA）
PADDLEOCR_VL_VLLM_AVAILABLE = importlib.util.find_spec("paddleocr_vl_vllm") is not None
if PADDLEOCR_VL_VLLM_AVAILABLE:
    logger.info("✅ PaddleOCR-VL-VLLM engine available")
else:
    logger.info("ℹ️  PaddleOCR-VL-VLLM not available (optional)")

# 检查 MinerU Pipeline 是否可用
MINERU_PIPELINE_AVAILABLE = importlib.util.find_spec("mineru_pipeline") is not None
if MINERU_PIPELINE_AVAILABLE:
    logger.info("✅ MinerU Pipeline engine available")
else:
    logger.info("ℹ️  MinerU Pipeline not available (optional)")

# 尝试导入 SenseVoice 音频处理
SENSEVOICE_AVAILABLE = importlib.util.find_spec("audio_engines") is not None
if SENSEVOICE_AVAILABLE:
    logger.info("✅ SenseVoice audio engine available")
else:
    logger.info("ℹ️  SenseVoice not available (optional)")

# 尝试导入视频处理引擎
VIDEO_ENGINE_AVAILABLE = importlib.util.find_spec("video_engines") is not None
if VIDEO_ENGINE_AVAILABLE:
    logger.info("✅ Video processing engine available")
else:
    logger.info("ℹ️  Video processing engine not available (optional)")

# 检查水印去除引擎是否可用（不要导入，避免初始化 CUDA）
WATERMARK_REMOVAL_AVAILABLE = importlib.util.find_spec("remove_watermark") is not None
if WATERMARK_REMOVAL_AVAILABLE:
    logger.info("✅ Watermark removal engine available")
else:
    logger.info("ℹ️  Watermark removal engine not available (optional)")

# 尝试导入格式引擎（专业领域格式支持）
try:
    from format_engines import FormatEngineRegistry, FASTAEngine, GenBankEngine

    # 注册所有引擎
    FormatEngineRegistry.register(FASTAEngine())
    FormatEngineRegistry.register(GenBankEngine())

    FORMAT_ENGINES_AVAILABLE = True
    logger.info("✅ Format engines available")
    logger.info(f"   Supported extensions: {', '.join(FormatEngineRegistry.get_supported_extensions())}")
except ImportError as e:
    FORMAT_ENGINES_AVAILABLE = False
    logger.info(f"ℹ️  Format engines not available (optional): {e}")


class MinerUWorkerAPI(ls.LitAPI):
    def __init__(
        self,
        paddleocr_vl_vllm_api_list=None,
        output_dir=None,
        poll_interval=0.5,
        enable_worker_loop=True,
        paddleocr_vl_vllm_engine_enabled=False,
    ):
        """
        初始化 API：直接在这里接收所有需要的参数
        """
        super().__init__()
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        default_output = project_root / "data" / "output"
        self.output_dir = output_dir or os.getenv("OUTPUT_PATH", str(default_output))
        self.poll_interval = poll_interval
        self.enable_worker_loop = enable_worker_loop
        self.paddleocr_vl_vllm_engine_enabled = paddleocr_vl_vllm_engine_enabled
        self.paddleocr_vl_vllm_api_list = paddleocr_vl_vllm_api_list or []
        ctx = multiprocessing.get_context("spawn")
        self._global_worker_counter = ctx.Value("i", 0)

    def setup(self, device):
        """
        初始化 Worker (每个 GPU 上调用一次)

        Args:
            device: 设备 ID (cuda:0, cuda:1, cpu 等)
        """
        ## 配置每个 Worker 的全局索引并尝试性分配self.paddleocr_vl_vllm_api
        with self._global_worker_counter.get_lock():
            my_global_index = self._global_worker_counter.value
            self._global_worker_counter.value += 1
        logger.info(f"🔢 [Init] I am Global Worker #{my_global_index} (on {device})")
        if self.paddleocr_vl_vllm_engine_enabled and len(self.paddleocr_vl_vllm_api_list) > 0:
            assigned_api = self.paddleocr_vl_vllm_api_list[my_global_index % len(self.paddleocr_vl_vllm_api_list)]
            self.paddleocr_vl_vllm_api = assigned_api
            logger.info(f"🔧 Worker #{my_global_index} assigned Paddle OCR VL API: {assigned_api}")
        else:
            self.paddleocr_vl_vllm_api = None
            logger.info(f"🔧 Worker #{my_global_index} assigned Paddle OCR VL API: None")

        # ============================================================================
        # 【关键】第一步：立即设置 CUDA_VISIBLE_DEVICES（必须在任何导入之前）
        # ============================================================================
        # LitServe 为每个 worker 进程分配不同的 device (cuda:0, cuda:1, ...)
        # 我们需要在导入任何 CUDA 库之前设置环境变量，实现进程级 GPU 隔离
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            # 【关键】设置 MinerU 的设备模式为 cuda:0
            # 因为设置了 CUDA_VISIBLE_DEVICES 后，进程只能看到一张卡（逻辑 ID 变为 0）
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
            logger.info(f"🎯 [GPU Isolation] Set CUDA_VISIBLE_DEVICES={gpu_id} (Physical GPU {gpu_id} → Logical GPU 0)")
            logger.info("🎯 [GPU Isolation] Set MINERU_DEVICE_MODE=cuda:0")

        import socket

        # 配置模型下载源（必须在 MinerU 初始化之前）
        # 从环境变量 MODEL_DOWNLOAD_SOURCE 读取配置
        # 支持: modelscope, huggingface, auto (默认)
        model_source = os.getenv("MODEL_DOWNLOAD_SOURCE", "auto").lower()

        if model_source in ["modelscope", "auto"]:
            # 尝试使用 ModelScope（优先）
            try:
                import importlib.util

                if importlib.util.find_spec("modelscope") is not None:
                    logger.info("📦 Model download source: ModelScope (国内推荐)")
                    logger.info("   Note: ModelScope automatically uses China mirror for faster downloads")
                else:
                    raise ImportError("modelscope not found")
            except ImportError:
                if model_source == "modelscope":
                    logger.warning("⚠️  ModelScope not available, falling back to HuggingFace")
                model_source = "huggingface"

        if model_source == "huggingface":
            # 配置 HuggingFace 镜像（从环境变量读取，默认使用国内镜像）
            hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
            logger.info(f"📦 Model download source: HuggingFace (via: {hf_endpoint})")
        elif model_source == "modelscope":
            ## 通过环境变量配置,来让模型从modelscope平台下载, 或者从modelscope的缓存目录加载
            os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
            logger.info("📦 Model download source: ModelScope")
        else:
            logger.warning(f"⚠️  Unknown model download source: {model_source}")

        self.device = device
        # 保存 accelerator 类型（从 device 字符串推断）
        # device 可能是 "cuda:0", "cuda:1", "cpu" 等
        if "cuda" in str(device):
            self.accelerator = "cuda"
            self.engine_device = "cuda:0"  # 引擎统一使用 cuda:0（因为已设置 CUDA_VISIBLE_DEVICES）
        else:
            self.accelerator = "cpu"
            self.engine_device = "cpu"  # CPU 模式

        logger.info(f"🎯 [Device] Accelerator: {self.accelerator}, Engine Device: {self.engine_device}")

        # 从类属性获取配置（由 start_litserve_workers 设置）
        # 默认使用共享输出目录（Docker 环境）
        project_root = Path(__file__).parent.parent
        default_output_path = project_root / "data" / "output"
        default_output = os.getenv("OUTPUT_PATH", str(default_output_path))
        self.output_dir = getattr(self.__class__, "_output_dir", default_output)
        self.poll_interval = getattr(self.__class__, "_poll_interval", 0.5)
        self.enable_worker_loop = getattr(self.__class__, "_enable_worker_loop", True)

        # ============================================================================
        # 第二步：现在可以安全地导入 MinerU 了（CUDA_VISIBLE_DEVICES 已设置）
        # ============================================================================
        global get_vram, clean_memory
        from mineru.utils.model_utils import get_vram, clean_memory

        # 配置 MinerU 的 VRAM 设置
        if os.getenv("MINERU_VIRTUAL_VRAM_SIZE", None) is None:
            device_mode = os.environ.get("MINERU_DEVICE_MODE", str(device))
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                try:
                    # 注意：get_vram 需要传入设备字符串（如 "cuda:0"）
                    vram = round(get_vram(device_mode))
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram)
                    logger.info(f"🎮 [MinerU VRAM] Detected: {vram}GB")
                except Exception as e:
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "8"  # 默认值
                    logger.warning(f"⚠️  Failed to detect VRAM, using default: 8GB ({e})")
            else:
                os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "1"
                logger.info("🎮 [MinerU VRAM] CPU mode, set to 1GB")

        # 验证 PyTorch CUDA 设置
        try:
            import torch

            if torch.cuda.is_available():
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
                device_count = torch.cuda.device_count()
                logger.info("✅ PyTorch CUDA verified:")
                logger.info(f"   CUDA_VISIBLE_DEVICES = {visible_devices}")
                logger.info(f"   torch.cuda.device_count() = {device_count}")
                if device_count == 1:
                    logger.info(f"   ✅ SUCCESS: Process isolated to 1 GPU (physical GPU {visible_devices})")
                else:
                    logger.warning(f"   ⚠️  WARNING: Expected 1 GPU but found {device_count}")
            else:
                logger.warning("⚠️  CUDA not available")
        except Exception as e:
            logger.warning(f"⚠️  Failed to verify PyTorch CUDA: {e}")

        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 初始化任务数据库（从环境变量读取，兼容 Docker 和本地）
        db_path_env = os.getenv("DATABASE_PATH")
        if db_path_env:
            db_path = Path(db_path_env).resolve()  # 使用 resolve() 转换为绝对路径
            logger.info(f"📊 Using DATABASE_PATH from environment: {db_path_env} -> {db_path}")
        else:
            # 默认路径（与 TaskDB 和 AuthDB 保持一致）
            project_root = Path(__file__).parent.parent
            default_db = project_root / "data" / "db" / "mineru_tianshu.db"
            db_path = default_db.resolve()
            logger.warning(f"⚠️  DATABASE_PATH not set, using default: {db_path}")

        # 确保数据库目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用绝对路径字符串传递给 TaskDB
        db_path_str = str(db_path.absolute())
        logger.info(f"📊 Database path (absolute): {db_path_str}")

        self.task_db = TaskDB(db_path_str)

        # 验证数据库连接并输出初始统计
        try:
            stats = self.task_db.get_queue_stats()
            logger.info(f"📊 Database initialized: {db_path} (exists: {db_path.exists()})")
            logger.info(f"📊 TaskDB.db_path: {self.task_db.db_path}")
            logger.info(f"📊 Initial queue stats: {stats}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database or get stats: {e}")
            logger.exception(e)

        # Worker 状态
        self.running = True
        self.current_task_id = None

        # 生成唯一的 worker_id: tianshu-{hostname}-{device}-{pid}
        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"tianshu-{hostname}-{device}-{pid}"
        # 子进程（setup 中）：

        # 初始化可选的处理引擎
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
        self.mineru_pipeline_engine = None  # 延迟加载
        self.paddleocr_vl_engine = None  # 延迟加载
        self.paddleocr_vl_vllm_engine = None  # 延迟加载
        self.sensevoice_engine = None  # 延迟加载
        self.video_engine = None  # 延迟加载
        self.watermark_handler = None  # 延迟加载

        logger.info("=" * 60)
        logger.info(f"🚀 Worker Setup: {self.worker_id}")
        logger.info("=" * 60)
        logger.info(f"📍 Device: {device}")
        logger.info(f"📂 Output Dir: {self.output_dir}")
        logger.info(f"🗃️  Database: {db_path}")
        logger.info(f"🔄 Worker Loop: {'Enabled' if self.enable_worker_loop else 'Disabled'}")
        if self.enable_worker_loop:
            logger.info(f"⏱️  Poll Interval: {self.poll_interval}s")
        logger.info("")

        # 打印可用的引擎
        logger.info("📦 Available Engines:")
        logger.info(f"   • MarkItDown: {'✅' if MARKITDOWN_AVAILABLE else '❌'}")
        logger.info(f"   • MinerU Pipeline: {'✅' if MINERU_PIPELINE_AVAILABLE else '❌'}")
        logger.info(f"   • PaddleOCR-VL: {'✅' if PADDLEOCR_VL_AVAILABLE else '❌'}")
        logger.info(f"   • SenseVoice: {'✅' if SENSEVOICE_AVAILABLE else '❌'}")
        logger.info(f"   • Video Engine: {'✅' if VIDEO_ENGINE_AVAILABLE else '❌'}")
        logger.info(f"   • Watermark Removal: {'✅' if WATERMARK_REMOVAL_AVAILABLE else '❌'}")
        logger.info(f"   • Format Engines: {'✅' if FORMAT_ENGINES_AVAILABLE else '❌'}")
        logger.info("")

        # 检测和初始化水印去除引擎（仅 CUDA）
        if WATERMARK_REMOVAL_AVAILABLE and "cuda" in str(device).lower():
            try:
                logger.info("🎨 Initializing watermark removal engine...")
                # 延迟导入，确保在 CUDA_VISIBLE_DEVICES 设置之后
                from remove_watermark.pdf_watermark_handler import PDFWatermarkHandler

                # 注意：由于在 setup() 中已设置 CUDA_VISIBLE_DEVICES，
                # 该进程只能看到一个 GPU（映射为 cuda:0）
                self.watermark_handler = PDFWatermarkHandler(device="cuda:0", use_lama=True)
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ Watermark removal engine initialized on cuda:0 (physical GPU {gpu_id})")
            except Exception as e:
                logger.error(f"❌ Failed to initialize watermark removal engine: {e}")
                self.watermark_handler = None

        logger.info("✅ Worker ready")
        logger.info(f"   LitServe Device: {device}")
        logger.info(f"   MinerU Device Mode: {os.environ.get('MINERU_DEVICE_MODE', 'auto')}")
        logger.info(f"   MinerU VRAM: {os.environ.get('MINERU_VIRTUAL_VRAM_SIZE', 'unknown')}GB")
        if "cuda" in str(device).lower():
            physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"   Physical GPU: {physical_gpu}")

        # 如果启用了 worker 循环，启动后台线程拉取任务
        if self.enable_worker_loop:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"🔄 Worker loop started (poll_interval={self.poll_interval}s)")
        else:
            logger.info("⏸️  Worker loop disabled, waiting for manual triggers")

    def _worker_loop(self):
        """
        Worker 后台循环：持续拉取任务并处理

        这个循环在后台线程中运行，不断检查是否有新任务
        一旦有任务，立即处理，处理完成后继续循环
        """
        logger.info(f"🔁 {self.worker_id} started task polling loop")

        # 记录初始诊断信息
        try:
            stats = self.task_db.get_queue_stats()
            logger.info(f"📊 Initial queue stats: {stats}")
            logger.info(f"🗃️  Database path: {self.task_db.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to get initial queue stats: {e}")

        loop_count = 0
        last_stats_log = 0
        stats_log_interval = 20  # 每20次循环输出一次统计信息（约10秒）

        while self.running:
            try:
                loop_count += 1

                # 拉取任务（原子操作，防止重复处理）
                task = self.task_db.get_next_task(worker_id=self.worker_id)

                if task:
                    task_id = task["task_id"]
                    self.current_task_id = task_id
                    logger.info(
                        f"📥 {self.worker_id} pulled task: {task_id} (file: {task.get('file_name', 'unknown')})"
                    )

                    try:
                        # 处理任务
                        self._process_task(task)
                        logger.info(f"✅ {self.worker_id} completed task: {task_id}")
                    except Exception as e:
                        logger.error(f"❌ {self.worker_id} failed task {task_id}: {e}")
                        logger.exception(e)
                    finally:
                        self.current_task_id = None
                else:
                    # 没有任务，空闲等待
                    # 定期输出统计信息以便诊断
                    if loop_count - last_stats_log >= stats_log_interval:
                        try:
                            stats = self.task_db.get_queue_stats()
                            pending = stats.get("pending", 0)
                            processing = stats.get("processing", 0)

                            if pending > 0:
                                logger.warning(
                                    f"⚠️  {self.worker_id} polling (loop #{loop_count}): "
                                    f"{pending} pending tasks found but not pulled! "
                                    f"Processing: {processing}, Completed: {stats.get('completed', 0)}, "
                                    f"Failed: {stats.get('failed', 0)}"
                                )
                            elif loop_count % 100 == 0:  # 每50秒（100次循环）输出一次
                                logger.info(
                                    f"💤 {self.worker_id} idle (loop #{loop_count}): "
                                    f"No pending tasks. Queue stats: {stats}"
                                )
                        except Exception as e:
                            logger.error(f"❌ Failed to get queue stats: {e}")

                        last_stats_log = loop_count

                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"❌ Worker loop error (loop #{loop_count}): {e}")
                logger.exception(e)
                time.sleep(self.poll_interval)

    def _process_task(self, task: dict):
        """
        处理单个任务

        Args:
            task: 任务字典（从数据库拉取）
        """
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}"))
        parent_task_id = task.get("parent_task_id")

        try:
            # 根据 backend 选择处理方式（从 task 字段读取，不是从 options 读取）
            backend = task.get("backend", "auto")

            # 检查文件扩展名
            file_ext = Path(file_path).suffix.lower()

            # 【新增】Office 转 PDF 预处理
            office_extensions = [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"]
            if file_ext in office_extensions and options.get("convert_office_to_pdf", False):
                logger.info(f"📄 [Preprocessing] Converting Office to PDF: {file_path}")
                try:
                    pdf_path = self._convert_office_to_pdf(file_path)

                    # 更新文件路径和扩展名
                    original_file_path = file_path
                    file_path = pdf_path
                    file_ext = ".pdf"

                    logger.info(f"✅ [Preprocessing] Office converted, continuing with PDF: {pdf_path}")
                    logger.info(f"   Original: {Path(original_file_path).name}")
                    logger.info(f"   Converted: {Path(pdf_path).name}")

                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Office to PDF conversion failed: {e}")
                    logger.warning(f"   Falling back to MarkItDown for: {file_path}")
                    # 转换失败，继续使用原文件（MarkItDown 处理）

            # 检查是否需要拆分 PDF（仅对非子任务的 PDF 进行判断）
            if file_ext == ".pdf" and not parent_task_id:
                if self._should_split_pdf(task_id, file_path, task, options):
                    # PDF 已被拆分，当前任务已转为父任务，直接返回
                    return

            # 0. 可选：预处理 - 去除水印（仅 PDF，作为预处理步骤）
            if file_ext == ".pdf" and options.get("remove_watermark", False) and self.watermark_handler:
                logger.info(f"🎨 [Preprocessing] Removing watermark from PDF: {file_path}")
                try:
                    cleaned_pdf_path = self._preprocess_remove_watermark(file_path, options)
                    file_path = str(cleaned_pdf_path)  # 使用去水印后的文件继续处理
                    logger.info(f"✅ [Preprocessing] Watermark removed, continuing with: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Watermark removal failed: {e}, continuing with original file")
                    # 继续使用原文件处理

            # 统一的引擎路由逻辑：优先使用用户指定的 backend，否则自动选择
            result = None  # 初始化 result

            # 1. 用户指定了音频引擎
            if backend == "sensevoice":
                if not SENSEVOICE_AVAILABLE:
                    raise ValueError("SenseVoice engine is not available")
                logger.info(f"🎤 Processing with SenseVoice: {file_path}")
                result = self._process_audio(file_path, options)

            # 3. 用户指定了视频引擎
            elif backend == "video":
                if not VIDEO_ENGINE_AVAILABLE:
                    raise ValueError("Video processing engine is not available")
                logger.info(f"🎬 Processing with video engine: {file_path}")
                result = self._process_video(file_path, options)

            # 4. 用户指定了 PaddleOCR-VL
            elif backend == "paddleocr-vl":
                if not PADDLEOCR_VL_AVAILABLE:
                    raise ValueError("PaddleOCR-VL engine is not available")
                logger.info(f"🔍 Processing with PaddleOCR-VL: {file_path}")
                result = self._process_with_paddleocr_vl(file_path, options)

            # 5. 用户指定了 PaddleOCR-VL-VLLM
            elif backend == "paddleocr-vl-vllm":
                if (
                    not PADDLEOCR_VL_VLLM_AVAILABLE
                    or not self.paddleocr_vl_vllm_engine_enabled
                    or len(self.paddleocr_vl_vllm_api_list) == 0
                ):
                    raise ValueError("PaddleOCR-VL-VLLM engine is not available")
                logger.info(f"🔍 Processing with PaddleOCR-VL-VLLM: {file_path}")
                result = self._process_with_paddleocr_vl_vllm(file_path, options)
            
            # 6. 用户指定了 MinerU 的某种模式 (pipeline, vlm, hybrid)
            elif backend in ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]:
                if not MINERU_PIPELINE_AVAILABLE:
                    raise ValueError(f"MinerU Pipeline engine is not available, cannot run {backend}")
                
                logger.info(f"🔧 Processing with MinerU ({backend}): {file_path}")
                
                # 将 backend 模式写入 options，传递给 Engine
                options["parse_mode"] = backend
                result = self._process_with_mineru(file_path, options)

            # 7. auto 模式：根据文件类型自动选择引擎
            elif backend == "auto":
                # 7.1 检查是否是专业格式（FASTA, GenBank 等）
                if FORMAT_ENGINES_AVAILABLE and FormatEngineRegistry.is_supported(file_path):
                    logger.info(f"🧬 [Auto] Processing with format engine: {file_path}")
                    result = self._process_with_format_engine(file_path, options)

                # 7.2 检查是否是音频文件
                elif file_ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"] and SENSEVOICE_AVAILABLE:
                    logger.info(f"🎤 [Auto] Processing audio file: {file_path}")
                    result = self._process_audio(file_path, options)

                # 7.3 检查是否是视频文件
                elif file_ext in [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"] and VIDEO_ENGINE_AVAILABLE:
                    logger.info(f"🎬 [Auto] Processing video file: {file_path}")
                    result = self._process_video(file_path, options)

                # 7.4 默认使用 MinerU Pipeline 处理 PDF/图片
                elif file_ext in [".pdf", ".png", ".jpg", ".jpeg"] and MINERU_PIPELINE_AVAILABLE:
                    logger.info(f"🔧 [Auto] Processing with MinerU Pipeline (Default): {file_path}")
                    # 默认使用 pipeline 模式
                    options["parse_mode"] = "pipeline" 
                    result = self._process_with_mineru(file_path, options)

                # 7.5 兜底：Office 文档/文本/HTML 使用 MarkItDown（如果可用）
                elif (
                    file_ext in [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".html", ".txt", ".csv"]
                    and self.markitdown
                ):
                    logger.info(f"📄 [Auto] Processing Office/Text file with MarkItDown: {file_path}")
                    result = self._process_with_markitdown(file_path)

                else:
                    # 没有合适的处理器
                    supported_formats = "PDF, PNG, JPG (MinerU/PaddleOCR), Audio (SenseVoice), Video, FASTA, GenBank"
                    if self.markitdown:
                        supported_formats += ", Office/Text (MarkItDown)"
                    raise ValueError(
                        f"Unsupported file type: file={file_path}, ext={file_ext}. "
                        f"Supported formats: {supported_formats}"
                    )

            else:
                # 8. 尝试使用格式引擎（用户明确指定了 fasta, genbank 等）
                if FORMAT_ENGINES_AVAILABLE:
                    engine = FormatEngineRegistry.get_engine(backend)
                    if engine is not None:
                        logger.info(f"🧬 Processing with format engine: {backend}")
                        result = self._process_with_format_engine(file_path, options, engine_name=backend)
                    else:
                        # 未知的 backend
                        raise ValueError(
                            f"Unknown backend: {backend}. "
                            f"Supported backends: auto, pipeline, vlm-auto-engine, hybrid-auto-engine, paddleocr-vl, sensevoice, video, fasta, genbank"
                        )
                else:
                    # 格式引擎不可用
                    raise ValueError(
                        f"Unknown backend: {backend}. "
                        f"Supported backends: auto, pipeline, vlm-auto-engine, hybrid-auto-engine, paddleocr-vl, sensevoice, video"
                    )

            # 检查 result 是否被正确赋值
            if result is None:
                raise ValueError(f"No result generated for backend: {backend}, file: {file_path}")

            # 更新任务状态为完成
            self.task_db.update_task_status(
                task_id=task_id,
                status="completed",
                result_path=result["result_path"],
                error_message=None,
            )

            # ... (后续代码不变) ...
