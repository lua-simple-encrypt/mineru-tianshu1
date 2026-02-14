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
import requests

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
            yield  # 什么都不做，直接让服务器启动

        def connect_mcp_server(self, *args, **kwargs):
            """空的 connect_mcp_server 方法，不做任何事情"""
            pass  # 什么都不做，跳过 MCP 初始化

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
    logger.warning("⚠️  markitdown not available, Office format parsing will be disabled")

# 检查 PaddleOCR-VL 是否可用（不要导入，避免初始化 CUDA）
PADDLEOCR_VL_AVAILABLE = importlib.util.find_spec("paddleocr_vl") is not None
if PADDLEOCR_VL_AVAILABLE:
    logger.info("✅ PaddleOCR-VL engine available")
else:
    logger.info("ℹ️  PaddleOCR-VL not available (optional)")

# 检查 PaddleOCR-VL-VLLM 是否可用（不要导入，避免初始化 CUDA）
PADDLEOCR_VL_VLLM_AVAILABLE = importlib.util.find_spec("paddleocr_vl_vllm") is not None
if PADDLEOCR_VL_VLLM_AVAILABLE:
    logger.info("✅ PaddleOCR-VL-VLLM engine available")
else:
    logger.info("ℹ️  PaddleOCR-VL-VLLM not available (optional)")

# 检查 MinerU Pipeline 是否可用
MINERU_PIPELINE_AVAILABLE = importlib.util.find_spec("mineru_pipeline") is not None
if MINERU_PIPELINE_AVAILABLE:
    logger.info("✅ MinerU Pipeline engine available")
else:
    logger.info("ℹ️  MinerU Pipeline not available (optional)")

# 尝试导入 SenseVoice 音频处理
SENSEVOICE_AVAILABLE = importlib.util.find_spec("audio_engines") is not None
if SENSEVOICE_AVAILABLE:
    logger.info("✅ SenseVoice audio engine available")
else:
    logger.info("ℹ️  SenseVoice not available (optional)")

# 尝试导入视频处理引擎
VIDEO_ENGINE_AVAILABLE = importlib.util.find_spec("video_engines") is not None
if VIDEO_ENGINE_AVAILABLE:
    logger.info("✅ Video processing engine available")
else:
    logger.info("ℹ️  Video processing engine not available (optional)")

# 检查水印去除引擎是否可用（不要导入，避免初始化 CUDA）
WATERMARK_REMOVAL_AVAILABLE = importlib.util.find_spec("remove_watermark") is not None
if WATERMARK_REMOVAL_AVAILABLE:
    logger.info("✅ Watermark removal engine available")
else:
    logger.info("ℹ️  Watermark removal engine not available (optional)")

# 尝试导入格式引擎（专业领域格式支持）
try:
    from format_engines import FormatEngineRegistry, FASTAEngine, GenBankEngine

    # 注册所有引擎
    FormatEngineRegistry.register(FASTAEngine())
    FormatEngineRegistry.register(GenBankEngine())

    FORMAT_ENGINES_AVAILABLE = True
    logger.info("✅ Format engines available")
    logger.info(f"   Supported extensions: {', '.join(FormatEngineRegistry.get_supported_extensions())}")
except ImportError as e:
    FORMAT_ENGINES_AVAILABLE = False
    logger.info(f"ℹ️  Format engines not available (optional): {e}")


# ==============================================================================
# VLLM Container Controller (互斥切换版 + Pickle 修复)
# ==============================================================================
class VLLMController:
    """管理 vLLM Docker 容器的互斥启动"""
    
    def __init__(self):
        # 不在 __init__ 中创建 client，确保对象是可序列化的 (Pickle Safe)
        pass

    def _get_client(self):
        """按需获取 Docker 客户端"""
        try:
            import docker
            # 连接到挂载的 /var/run/docker.sock
            return docker.from_env()
        except Exception as e:
            logger.warning(f"⚠️  Docker client init failed: {e}")
            return None

    def ensure_service(self, target_container: str, conflict_container: str, health_url: str, timeout: int = 300):
        """
        确保目标容器运行，并关闭冲突容器 (互斥逻辑)
        
        Args:
            target_container: 需要运行的容器名
            conflict_container: 需要关闭的互斥容器名
            health_url: 目标容器的健康检查地址
            timeout: 超时时间
        """
        client = self._get_client()
        if not client:
            return
        
        try:
            # 1. 检查目标容器是否已经在运行
            try:
                target = client.containers.get(target_container)
                if target.status == 'running':
                    # 如果已经在运行，直接返回，无需操作
                    logger.info(f"✅ Target service {target_container} is already running.")
                    return
            except Exception as e:
                # 如果找不到容器，说明没创建，提示用户
                logger.error(f"❌ Container {target_container} not found. Please ensure it is created (e.g. docker compose up --no-start).")
                raise e

            # 2. 停止冲突容器 (释放显存)
            try:
                conflict = client.containers.get(conflict_container)
                if conflict.status == 'running':
                    logger.info(f"🛑 Stopping conflicting service {conflict_container} to free VRAM...")
                    conflict.stop()
                    logger.info(f"✅ Service {conflict_container} stopped.")
            except Exception:
                # 冲突容器可能不存在或已停止，忽略
                pass

            # 3. 启动目标容器
            logger.info(f"🚀 Starting service {target_container} (Cold Start)...")
            target.start()

            # 4. 等待健康检查
            self._wait_for_health(health_url, timeout)
            
        finally:
            try:
                client.close()
            except:
                pass

    def _wait_for_health(self, url: str, timeout: int):
        """轮询健康检查接口"""
        start_time = time.time()
        logger.info(f"⏳ Waiting for service at {url} (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                # 显式使用 host.docker.internal 或者是 Docker DNS 名
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ Service is ready: {url}")
                    return
            except Exception:
                pass
            
            time.sleep(2) # 每2秒重试一次
        
        raise TimeoutError(f"Service at {url} did not become ready in {timeout} seconds")


class MinerUWorkerAPI(ls.LitAPI):
    def __init__(
        self,
        paddleocr_vl_vllm_api_list=None,
        mineru_vllm_api_list=None,
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
        self.mineru_vllm_api_list = mineru_vllm_api_list or []  # 保存 MinerU API 列表
        
        ctx = multiprocessing.get_context("spawn")
        self._global_worker_counter = ctx.Value("i", 0)

        # 【关键修改】在 __init__ 中直接初始化 VLLMController
        # 因为现在的 VLLMController 不持有不可序列化的 client 对象，所以是安全的
        self.vllm_controller = VLLMController()

    def setup(self, device):
        """
        初始化 Worker (每个 GPU 上调用一次)

        Args:
            device: 设备 ID (cuda:0, cuda:1, cpu 等)
        """
        ## 配置每个 Worker 的全局索引并尝试性分配 API
        with self._global_worker_counter.get_lock():
            my_global_index = self._global_worker_counter.value
            self._global_worker_counter.value += 1
        logger.info(f"🔢 [Init] I am Global Worker #{my_global_index} (on {device})")
        
        # 1. 分配 PaddleOCR VLLM API
        if self.paddleocr_vl_vllm_engine_enabled and len(self.paddleocr_vl_vllm_api_list) > 0:
            assigned_api = self.paddleocr_vl_vllm_api_list[my_global_index % len(self.paddleocr_vl_vllm_api_list)]
            self.paddleocr_vl_vllm_api = assigned_api
            logger.info(f"🔧 Worker #{my_global_index} assigned Paddle OCR VL API: {assigned_api}")
        else:
            self.paddleocr_vl_vllm_api = None
            logger.info(f"🔧 Worker #{my_global_index} assigned Paddle OCR VL API: None")

        # 2. 分配 MinerU VLLM API (新增)
        if len(self.mineru_vllm_api_list) > 0:
            assigned_mineru_api = self.mineru_vllm_api_list[my_global_index % len(self.mineru_vllm_api_list)]
            self.mineru_vllm_api = assigned_mineru_api
            logger.info(f"🔧 Worker #{my_global_index} assigned MinerU VLLM API: {assigned_mineru_api}")
        else:
            self.mineru_vllm_api = None
            logger.info(f"🔧 Worker #{my_global_index} assigned MinerU VLLM API: None")

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
                    logger.info("   Note: ModelScope automatically uses China mirror for faster downloads")
                else:
                    raise ImportError("modelscope not found")
            except ImportError:
                if model_source == "modelscope":
                    logger.warning("⚠️  ModelScope not available, falling back to HuggingFace")
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
            logger.warning(f"⚠️  Unknown model download source: {model_source}")

        self.device = device
        # 保存 accelerator 类型（从 device 字符串推断）
        # device 可能是 "cuda:0", "cuda:1", "cpu" 等
        if "cuda" in str(device):
            self.accelerator = "cuda"
            self.engine_device = "cuda:0"  # 引擎统一使用 cuda:0（因为已设置 CUDA_VISIBLE_DEVICES）
        else:
            self.accelerator = "cpu"
            self.engine_device = "cpu"  # CPU 模式

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
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "8"  # 默认值
                    logger.warning(f"⚠️  Failed to detect VRAM, using default: 8GB ({e})")
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
                logger.info(f"   CUDA_VISIBLE_DEVICES = {visible_devices}")
                logger.info(f"   torch.cuda.device_count() = {device_count}")
                if device_count == 1:
                    logger.info(f"   ✅ SUCCESS: Process isolated to 1 GPU (physical GPU {visible_devices})")
                else:
                    logger.warning(f"   ⚠️  WARNING: Expected 1 GPU but found {device_count}")
            else:
                logger.warning("⚠️  CUDA not available")
        except Exception as e:
            logger.warning(f"⚠️  Failed to verify PyTorch CUDA: {e}")

        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 初始化任务数据库（从环境变量读取，兼容 Docker 和本地）
        db_path_env = os.getenv("DATABASE_PATH")
        if db_path_env:
            db_path = Path(db_path_env).resolve()  # 使用 resolve() 转换为绝对路径
            logger.info(f"📊 Using DATABASE_PATH from environment: {db_path_env} -> {db_path}")
        else:
            # 默认路径（与 TaskDB 和 AuthDB 保持一致）
            project_root = Path(__file__).parent.parent
            default_db = project_root / "data" / "db" / "mineru_tianshu.db"
            db_path = default_db.resolve()
            logger.warning(f"⚠️  DATABASE_PATH not set, using default: {db_path}")

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
        self.mineru_pipeline_engine = None  # 延迟加载
        self.paddleocr_vl_engine = None  # 延迟加载
        self.paddleocr_vl_vllm_engine = None  # 延迟加载
        self.sensevoice_engine = None  # 延迟加载
        self.video_engine = None  # 延迟加载
        self.watermark_handler = None  # 延迟加载

        logger.info("=" * 60)
        logger.info(f"🚀 Worker Setup: {self.worker_id}")
        logger.info("=" * 60)
        logger.info(f"📍 Device: {device}")
        logger.info(f"📂 Output Dir: {self.output_dir}")
        logger.info(f"🗃️  Database: {db_path}")
        logger.info(f"🔄 Worker Loop: {'Enabled' if self.enable_worker_loop else 'Disabled'}")
        if self.enable_worker_loop:
            logger.info(f"⏱️  Poll Interval: {self.poll_interval}s")
        logger.info("")

        # 打印可用的引擎
        logger.info("📦 Available Engines:")
        logger.info(f"   • MarkItDown: {'✅' if MARKITDOWN_AVAILABLE else '❌'}")
        logger.info(f"   • MinerU Pipeline: {'✅' if MINERU_PIPELINE_AVAILABLE else '❌'}")
        logger.info(f"   • PaddleOCR-VL: {'✅' if PADDLEOCR_VL_AVAILABLE else '❌'}")
        logger.info(f"   • SenseVoice: {'✅' if SENSEVOICE_AVAILABLE else '❌'}")
        logger.info(f"   • Video Engine: {'✅' if VIDEO_ENGINE_AVAILABLE else '❌'}")
        logger.info(f"   • Watermark Removal: {'✅' if WATERMARK_REMOVAL_AVAILABLE else '❌'}")
        logger.info(f"   • Format Engines: {'✅' if FORMAT_ENGINES_AVAILABLE else '❌'}")
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
        logger.info(f"   LitServe Device: {device}")
        logger.info(f"   MinerU Device Mode: {os.environ.get('MINERU_DEVICE_MODE', 'auto')}")
        logger.info(f"   MinerU VRAM: {os.environ.get('MINERU_VIRTUAL_VRAM_SIZE', 'unknown')}GB")
        if "cuda" in str(device).lower():
            physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"   Physical GPU: {physical_gpu}")

        # 如果启用了 worker 循环，启动后台线程拉取任务
        if self.enable_worker_loop:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"🔄 Worker loop started (poll_interval={self.poll_interval}s)")
        else:
            logger.info("⏸️  Worker loop disabled, waiting for manual triggers")

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
            logger.info(f"🗃️  Database path: {self.task_db.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to get initial queue stats: {e}")

        loop_count = 0
        last_stats_log = 0
        stats_log_interval = 20  # 每20次循环输出一次统计信息（约10秒）

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
                                    f"⚠️  {self.worker_id} polling (loop #{loop_count}): "
                                    f"{pending} pending tasks found but not pulled! "
                                    f"Processing: {processing}, Completed: {stats.get('completed', 0)}, "
                                    f"Failed: {stats.get('failed', 0)}"
                                )
                            elif loop_count % 100 == 0:  # 每50秒（100次循环）输出一次
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
        处理单个任务 (集成互斥启动逻辑)

        Args:
            task: 任务字典（从数据库拉取）
        """
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}"))
        parent_task_id = task.get("parent_task_id")
        backend = task.get("backend", "auto")
        
        try:
            # 1. 智能服务切换逻辑
            paddle_container = "tianshu-vllm-paddleocr"
            mineru_container = "tianshu-vllm-mineru"
            
            # 如果是 PaddleOCR 任务
            if backend == "paddleocr-vl-vllm" and self.paddleocr_vl_vllm_api:
                base = self.paddleocr_vl_vllm_api.replace("/v1", "")
                health = f"{base}/health"
                # 确保 Paddle 运行，关闭 MinerU
                self.vllm_controller.ensure_service(paddle_container, mineru_container, health)
                
            # 如果是 MinerU 任务 (vlm/hybrid local 模式)
            # 注意: remote client 模式不需要启动本地容器
            elif backend in ["vlm-auto-engine", "hybrid-auto-engine"] and self.mineru_vllm_api:
                base = self.mineru_vllm_api.replace("/v1", "")
                health = f"{base}/health"
                # 确保 MinerU 运行，关闭 Paddle
                self.vllm_controller.ensure_service(mineru_container, paddle_container, health)

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
                    logger.info(f"   Original: {Path(original_file_path).name}")
                    logger.info(f"   Converted: {Path(pdf_path).name}")

                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Office to PDF conversion failed: {e}")
                    logger.warning(f"   Falling back to MarkItDown for: {file_path}")
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
                    file_path = str(cleaned_pdf_path)  # 使用去水印后的文件继续处理
                    logger.info(f"✅ [Preprocessing] Watermark removed, continuing with: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Watermark removal failed: {e}, continuing with original file")
                    # 继续使用原文件处理

            # 统一的引擎路由逻辑：优先使用用户指定的 backend，否则自动选择
            result = None  # 初始化 result

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
            
            # 6. 用户指定了 MinerU 的某种模式 (pipeline, vlm-*, hybrid-*)
            elif "pipeline" in backend or "vlm-" in backend or "hybrid-" in backend:
                if not MINERU_PIPELINE_AVAILABLE:
                    raise ValueError(f"MinerU Pipeline engine is not available, cannot run {backend}")
                
                logger.info(f"🔧 Processing with MinerU ({backend}): {file_path}")
                
                # 将 backend 模式写入 options，传递给 Engine
                options["parse_mode"] = backend  # 【关键】确保 parse_mode 正确传递
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
                            f"Supported backends: auto, pipeline, vlm-*, hybrid-*, paddleocr-vl, sensevoice, video, fasta, genbank"
                        )
                else:
                    # 格式引擎不可用
                    raise ValueError(
                        f"Unknown backend: {backend}. "
                        f"Supported backends: auto, pipeline, vlm-*, hybrid-*, paddleocr-vl, sensevoice, video"
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

            # 如果是子任务,检查是否需要触发合并
            if parent_task_id:
                parent_id_to_merge = self.task_db.on_child_task_completed(task_id)

                if parent_id_to_merge:
                    # 所有子任务完成,执行合并
                    logger.info(f"🔀 All subtasks completed, merging results for parent task {parent_id_to_merge}")
                    try:
                        self._merge_parent_task_results(parent_id_to_merge)
                    except Exception as merge_error:
                        logger.error(f"❌ Failed to merge parent task {parent_id_to_merge}: {merge_error}")
                        # 标记父任务为失败
                        self.task_db.update_task_status(
                            parent_id_to_merge, "failed", error_message=f"Merge failed: {merge_error}"
                        )

            # 清理显存（如果是 GPU）
            if "cuda" in str(self.device).lower():
                clean_memory()

        except Exception as e:
            # 更新任务状态为失败
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.task_db.update_task_status(task_id=task_id, status="failed", result_path=None, error_message=error_msg)

            # 如果是子任务失败,标记父任务失败
            if parent_task_id:
                self.task_db.on_child_task_failed(task_id, error_msg)

            raise

    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        """
        使用 MinerU 处理文档

        注意：
        - MinerU 的 do_parse 只接受 PDF 格式，图片需要先转换为 PDF
        - CUDA_VISIBLE_DEVICES 已在 setup() 阶段设置，MinerU 会自动使用正确的 GPU
        """
        # 延迟加载 MinerU Pipeline（单例模式）
        if self.mineru_pipeline_engine is None:
            from mineru_pipeline import MinerUPipelineEngine

            # 使用动态设备选择（支持 CPU/CUDA）
            # 注意：CUDA 模式下已在 setup() 中设置 CUDA_VISIBLE_DEVICES，
            # 该进程只能看到一个 GPU（映射为 cuda:0）
            self.mineru_pipeline_engine = MinerUPipelineEngine(
                device=self.engine_device,
                vlm_api_base=self.mineru_vllm_api  # 传入 MinerU API 地址 (本地服务)
            )
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ MinerU Pipeline engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ MinerU Pipeline engine loaded on CPU")

        # 设置输出目录
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否是远程模式
        backend = options.get("parse_mode", "pipeline")
        if "http-client" in backend:
            # 如果是客户端模式，必须有 server_url
            if not options.get("server_url"):
                logger.warning(f"⚠️  Remote backend {backend} selected but no server_url provided in options.")
                # 尝试回退到默认本地服务（如果有）
                if self.mineru_vllm_api:
                     options["server_url"] = self.mineru_vllm_api.replace("/v1", "") # 去掉 /v1
                     logger.info(f"   Using default local server: {options['server_url']}")

        # 处理文件
        result = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)

        # 规范化输出（统一文件名和目录结构）
        # 注意：result["result_path"] 是实际包含 md 文件的目录（例如 {output_dir}/{file_name}/auto/）
        # 我们需要在这个result["result_path"] 上运行 normalize_output
        actual_output_dir = Path(result["result_path"])
        normalize_output(actual_output_dir)

        # MinerU Pipeline 返回结构：
        return {
            "result_path": result["result_path"],
            "content": result["markdown"],
            "json_path": result.get("json_path"),
            "json_content": result.get("json_content"),
        }

    def _process_with_markitdown(self, file_path: str) -> dict:
        """使用 MarkItDown 处理 Office 文档（增强版：支持 DOCX 图片提取）"""
        if not self.markitdown:
            raise RuntimeError("MarkItDown is not available")

        # 创建输出目录（与其他引擎保持一致）
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理文件：提取文本
        result = self.markitdown.convert(file_path)
        markdown_content = result.text_content

        # 如果是 DOCX 文件，提取嵌入的图片
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".docx":
            try:
                from utils.docx_image_extractor import extract_images_from_docx, append_images_to_markdown

                # 提取图片到 images 目录
                images_dir = output_dir / "images"
                images = extract_images_from_docx(file_path, str(images_dir))

                # 如果有图片，将图片引用添加到 Markdown
                if images:
                    markdown_content = append_images_to_markdown(markdown_content, images)
                    logger.info(f"🖼️  Extracted {len(images)} images from DOCX")

            except Exception as e:
                logger.warning(f"⚠️  Failed to extract images from DOCX: {e}")
                # 继续处理，不影响文本提取

        # 保存结果到目录中
        output_file = output_dir / f"{Path(file_path).stem}_markitdown.md"
        output_file.write_text(markdown_content, encoding="utf-8")

        # 规范化输出（统一文件名和目录结构）
        normalize_output(output_dir)

        # 返回目录路径（与其他引擎保持一致）
        return {"result_path": str(output_dir), "content": markdown_content}

    def _convert_office_to_pdf(self, file_path: str) -> str:
        """
        使用 LibreOffice 将 Office 文件转换为 PDF

        Args:
            file_path: Office 文件路径

        Returns:
            转换后的 PDF 文件路径

        Raises:
            RuntimeError: 转换失败时抛出
        """
        import subprocess
        import shutil
        import tempfile
        from pathlib import Path

        input_file = Path(file_path)
        final_output_dir = input_file.parent

        # 最终输出文件名
        final_pdf_file = final_output_dir / f"{input_file.stem}.pdf"

        # 如果已存在同名 PDF，先删除
        if final_pdf_file.exists():
            final_pdf_file.unlink()

        logger.info(f"🔄 Converting Office to PDF: {input_file.name}")

        try:
            # 使用 /tmp 作为临时目录（避免 Docker 挂载卷写入问题）
            with tempfile.TemporaryDirectory(prefix="libreoffice_") as temp_dir:
                temp_dir_path = Path(temp_dir)

                # 复制输入文件到临时目录
                temp_input = temp_dir_path / input_file.name
                shutil.copy2(input_file, temp_input)

                # 在临时目录执行转换
                cmd = [
                    "libreoffice",
                    "--headless",  # 无界面模式
                    "--convert-to",
                    "pdf",  # 转换为 PDF
                    "--outdir",
                    str(temp_dir_path),  # 输出到临时目录
                    str(temp_input),  # 输入文件
                ]

                # 执行转换（超时 120 秒）
                result = subprocess.run(cmd, check=True, timeout=120, capture_output=True, text=True)

                # 临时输出文件路径
                temp_pdf = temp_dir_path / f"{input_file.stem}.pdf"

                # 验证输出文件是否存在
                if not temp_pdf.exists():
                    stderr_output = result.stderr if result.stderr else "No error output"
                    raise RuntimeError(
                        f"LibreOffice conversion failed: output file not found: {temp_pdf}\nstderr: {stderr_output}"
                    )

                # 移动转换后的 PDF 到最终目录
                shutil.move(str(temp_pdf), str(final_pdf_file))

                logger.info(
                    f"✅ Office converted to PDF: {final_pdf_file.name} ({final_pdf_file.stat().st_size / 1024:.1f} KB)"
                )

                return str(final_pdf_file)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"LibreOffice conversion timeout (>120s): {input_file.name}")
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr if e.stderr else "No error output"
            raise RuntimeError(f"LibreOffice conversion failed: {stderr_output}")
        except Exception as e:
            raise RuntimeError(f"Office to PDF conversion error: {e}")

    def _process_with_paddleocr_vl(self, file_path: str, options: dict) -> dict:
        """使用 PaddleOCR-VL 处理图片或 PDF"""
        # 检查加速器类型（PaddleOCR-VL 仅支持 GPU）
        if self.accelerator == "cpu":
            raise RuntimeError(
                "PaddleOCR-VL requires GPU and is not supported in CPU mode. "
                "Please use 'mineru' or 'markitdown' backend instead."
            )

        # 延迟加载 PaddleOCR-VL（单例模式）
        if self.paddleocr_vl_engine is None:
            from paddleocr_vl import PaddleOCRVLEngine

            # 注意：由于在 setup() 中已设置 CUDA_VISIBLE_DEVICES，
            # 该进程只能看到一个 GPU（映射为 cuda:0）
            # 【关键修改】指定模型名称，匹配本地下载的目录
            self.paddleocr_vl_engine = PaddleOCRVLEngine(device="cuda:0", model_name="PaddleOCR-VL-1.5-0.9B")
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"✅ PaddleOCR-VL engine loaded on cuda:0 (physical GPU {gpu_id})")

        # 设置输出目录
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理文件（parse 方法需要 output_path）
        result = self.paddleocr_vl_engine.parse(file_path, output_path=str(output_dir))

        # 规范化输出（统一文件名和目录结构）
        normalize_output(output_dir)

        # 返回结果
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_with_paddleocr_vl_vllm(self, file_path: str, options: dict) -> dict:
        """使用 PaddleOCR-VL VLLM 处理图片或 PDF"""
        # 检查加速器类型（PaddleOCR-VL VLLM 仅支持 GPU）
        if self.accelerator == "cpu":
            raise RuntimeError(
                "PaddleOCR-VL VLLM requires GPU and is not supported in CPU mode. "
                "Please use 'mineru' or 'markitdown' backend instead."
            )

        # 延迟加载 PaddleOCR-VL（单例模式）
        if self.paddleocr_vl_vllm_engine is None:
            from paddleocr_vl_vllm import PaddleOCRVLVLLMEngine

            # 注意：由于在 setup() 中已设置 CUDA_VISIBLE_DEVICES，
            # 该进程只能看到一个 GPU（映射为 cuda:0）
            # 【关键修改】指定模型名称
            self.paddleocr_vl_vllm_engine = PaddleOCRVLVLLMEngine(
                device="cuda:0", 
                vllm_api_base=self.paddleocr_vl_vllm_api,
                model_name="PaddleOCR-VL-1.5-0.9B"
            )
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"✅ PaddleOCR-VL VLLM engine loaded on cuda:0 (physical GPU {gpu_id})")

        # 设置输出目录
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理文件（parse 方法需要 output_path）
        result = self.paddleocr_vl_vllm_engine.parse(file_path, output_path=str(output_dir))

        # 规范化输出（统一文件名和目录结构）
        normalize_output(output_dir, handle_method="paddleocr-vl")

        # 返回结果
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_audio(self, file_path: str, options: dict) -> dict:
        """使用 SenseVoice 处理音频文件"""
        # 延迟加载 SenseVoice（单例模式）
        if self.sensevoice_engine is None:
            from audio_engines import SenseVoiceEngine

            # 使用动态设备选择（支持 CPU/CUDA）
            # 注意：CUDA 模式下已在 setup() 中设置 CUDA_VISIBLE_DEVICES，
            # 该进程只能看到一个 GPU（映射为 cuda:0）
            self.sensevoice_engine = SenseVoiceEngine(device=self.engine_device)
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ SenseVoice engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ SenseVoice engine loaded on CPU")

        # 设置输出目录
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理音频（parse 方法需要 output_path 参数）
        result = self.sensevoice_engine.parse(
            audio_path=file_path,
            output_path=str(output_dir),
            language=options.get("lang", "auto"),
            use_itn=options.get("use_itn", True),
            enable_speaker_diarization=options.get("enable_speaker_diarization", False),  # 从 API 参数控制
        )

        # 规范化输出（统一文件名和目录结构）
        normalize_output(output_dir)

        # SenseVoice 返回结构：
        # {
        #   "success": True,
        #   "output_path": str,
        #   "markdown": str,
        #   "markdown_file": str,
        #   "json_file": str,
        #   "json_data": dict,
        #   "result": dict
        # }
        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_video(self, file_path: str, options: dict) -> dict:
        """使用视频处理引擎处理视频文件"""
        # 延迟加载视频引擎（单例模式）
        if self.video_engine is None:
            from video_engines import VideoProcessingEngine

            # 使用动态设备选择（支持 CPU/CUDA）
            # 注意：CUDA 模式下已在 setup() 中设置 CUDA_VISIBLE_DEVICES，
            # 该进程只能看到一个 GPU（映射为 cuda:0）
            self.video_engine = VideoProcessingEngine(device=self.engine_device)
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ Video processing engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ Video processing engine loaded on CPU")

        # 创建输出目录（与其他引擎保持一致）
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理视频
        result = self.video_engine.parse(
            video_path=file_path,
            output_path=str(output_dir),
            language=options.get("lang", "auto"),
            use_itn=options.get("use_itn", True),
            keep_audio=options.get("keep_audio", False),
            enable_keyframe_ocr=options.get("enable_keyframe_ocr", False),
            ocr_backend=options.get("ocr_backend", "paddleocr-vl"),
            keep_keyframes=options.get("keep_keyframes", False),
        )

        # 保存结果（Markdown 格式）
        output_file = output_dir / f"{Path(file_path).stem}_video_analysis.md"
        output_file.write_text(result["markdown"], encoding="utf-8")

        # 规范化输出（统一文件名和目录结构）
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": result["markdown"]}

    def _preprocess_remove_watermark(self, file_path: str, options: dict) -> Path:
        """
        预处理：去除 PDF 水印

        这是一个可选的预处理步骤，去除水印后的文件会被后续的解析引擎处理

        返回：
            去除水印后的 PDF 路径

        支持的 options 参数：
            - auto_detect: 是否自动检测 PDF 类型（默认 True）
            - force_scanned: 强制使用扫描件模式（默认 False）
            - remove_text: 是否删除文本对象（可编辑 PDF，默认 True）
            - remove_images: 是否删除图片对象（可编辑 PDF，默认 True）
            - remove_annotations: 是否删除注释（可编辑 PDF，默认 True）
            - keywords: 文本关键词列表（可编辑 PDF，只删除包含这些关键词的文本）
            - dpi: 转换分辨率（扫描件 PDF，默认 200）
            - conf_threshold: YOLO 置信度阈值（扫描件 PDF，默认 0.35）
            - dilation: 掩码膨胀（扫描件 PDF，默认 10）
        """
        if not self.watermark_handler:
            raise RuntimeError("Watermark removal is not available (CUDA required)")

        # 设置输出路径
        output_file = Path(self.output_dir) / f"{Path(file_path).stem}_no_watermark.pdf"

        # 构建参数字典（只传递实际提供的参数）
        kwargs = {}

        # 通用参数
        if "auto_detect" in options:
            kwargs["auto_detect"] = options["auto_detect"]
        if "force_scanned" in options:
            kwargs["force_scanned"] = options["force_scanned"]

        # 可编辑 PDF 参数
        if "remove_text" in options:
            kwargs["remove_text"] = options["remove_text"]
        if "remove_images" in options:
            kwargs["remove_images"] = options["remove_images"]
        if "remove_annotations" in options:
            kwargs["remove_annotations"] = options["remove_annotations"]
        if "watermark_keywords" in options:
            kwargs["keywords"] = options["watermark_keywords"]

        # 扫描件 PDF 参数
        if "watermark_dpi" in options:
            kwargs["dpi"] = options["watermark_dpi"]
        if "watermark_conf_threshold" in options:
            kwargs["conf_threshold"] = options["watermark_conf_threshold"]
        if "watermark_dilation" in options:
            kwargs["dilation"] = options["watermark_dilation"]

        # 去除水印（返回输出路径）
        cleaned_pdf_path = self.watermark_handler.remove_watermark(
            input_path=file_path, output_path=str(output_file), **kwargs
        )

        return cleaned_pdf_path

    def _should_split_pdf(self, task_id: str, file_path: str, task: dict, options: dict) -> bool:
        """
        判断 PDF 是否需要拆分，如果需要则执行拆分

        Args:
            task_id: 任务ID
            file_path: PDF 文件路径
            task: 任务字典
            options: 处理选项

        Returns:
            bool: True 表示已拆分，False 表示不需要拆分
        """
        from utils.pdf_utils import get_pdf_page_count, split_pdf_file

        # 读取配置
        pdf_split_enabled = os.getenv("PDF_SPLIT_ENABLED", "true").lower() == "true"
        if not pdf_split_enabled:
            return False

        pdf_split_threshold = int(os.getenv("PDF_SPLIT_THRESHOLD_PAGES", "500"))
        pdf_split_chunk_size = int(os.getenv("PDF_SPLIT_CHUNK_SIZE", "500"))

        try:
            # 快速读取 PDF 页数（只读元数据）
            page_count = get_pdf_page_count(Path(file_path))
            logger.info(f"📄 PDF has {page_count} pages (threshold: {pdf_split_threshold})")

            # 判断是否需要拆分
            if page_count <= pdf_split_threshold:
                return False

            logger.info(
                f"🔀 Large PDF detected ({page_count} pages), splitting into chunks of {pdf_split_chunk_size} pages"
            )

            # 将当前任务转为父任务
            self.task_db.convert_to_parent_task(task_id, child_count=0)

            # 拆分 PDF 文件
            split_dir = Path(self.output_dir) / "splits" / task_id
            split_dir.mkdir(parents=True, exist_ok=True)

            chunks = split_pdf_file(
                pdf_path=Path(file_path),
                output_dir=split_dir,
                chunk_size=pdf_split_chunk_size,
                parent_task_id=task_id,
            )

            logger.info(f"✂️  PDF split into {len(chunks)} chunks")

            # 为每个分块创建子任务
            backend = task.get("backend", "auto")
            priority = task.get("priority", 0)
            user_id = task.get("user_id")

            for chunk_info in chunks:
                # 复制选项并添加分块信息
                chunk_options = options.copy()
                chunk_options["chunk_info"] = {
                    "start_page": chunk_info["start_page"],
                    "end_page": chunk_info["end_page"],
                    "page_count": chunk_info["page_count"],
                }

                # 创建子任务
                child_task_id = self.task_db.create_child_task(
                    parent_task_id=task_id,
                    file_name=f"{Path(file_path).stem}_pages_{chunk_info['start_page']}-{chunk_info['end_page']}.pdf",
                    file_path=chunk_info["path"],
                    backend=backend,
                    options=chunk_options,
                    priority=priority,
                    user_id=user_id,
                )

                logger.info(
                    f"  ✅ Created subtask {child_task_id}: pages {chunk_info['start_page']}-{chunk_info['end_page']}"
                )

            # 更新父任务的子任务数量
            self.task_db.convert_to_parent_task(task_id, child_count=len(chunks))

            logger.info(f"🎉 Large PDF split complete: {len(chunks)} subtasks created for parent task {task_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to split PDF: {e}")
            logger.warning("⚠️  Falling back to processing as single task")
            return False

    def _merge_parent_task_results(self, parent_task_id: str):
        """
        合并父任务的所有子任务结果

        Args:
            parent_task_id: 父任务ID
        """
        try:
            # 获取父任务和所有子任务
            parent_task = self.task_db.get_task_with_children(parent_task_id)

            if not parent_task:
                raise ValueError(f"Parent task {parent_task_id} not found")

            children = parent_task.get("children", [])

            if not children:
                raise ValueError(f"No child tasks found for parent {parent_task_id}")

            # 按页码排序子任务
            children.sort(key=lambda x: json.loads(x.get("options", "{}")).get("chunk_info", {}).get("start_page", 0))

            logger.info(f"🔀 Merging {len(children)} subtask results for parent task {parent_task_id}")

            # 创建父任务输出目录
            parent_output_dir = Path(self.output_dir) / Path(parent_task["file_path"]).stem
            parent_output_dir.mkdir(parents=True, exist_ok=True)

            # 合并 Markdown
            markdown_parts = []
            json_pages = []
            has_json = False

            for idx, child in enumerate(children):
                if child["status"] != "completed":
                    logger.warning(f"⚠️  Child task {child['task_id']} not completed (status: {child['status']})")
                    continue

                result_dir = Path(child["result_path"])
                chunk_info = json.loads(child.get("options", "{}")).get("chunk_info", {})

                # 读取 Markdown
                md_files = list(result_dir.rglob("*.md"))
                if md_files:
                    md_file = None
                    for f in md_files:
                        if f.name == "result.md":
                            md_file = f
                            break
                    if not md_file:
                        md_file = md_files[0]

                    content = md_file.read_text(encoding="utf-8")

                    # 添加分页标记
                    if chunk_info:
                        markdown_parts.append(
                            f"\n\n\n\n"
                        )
                    markdown_parts.append(content)

                    logger.info(
                        f"   ✅ Merged chunk {idx + 1}/{len(children)}: "
                        f"pages {chunk_info.get('start_page', '?')}-{chunk_info.get('end_page', '?')}"
                    )

                # 读取 JSON (如果有)
                json_files = [
                    f
                    for f in result_dir.rglob("*.json")
                    if f.name in ["content.json", "result.json"] or "_content_list.json" in f.name
                ]

                if json_files:
                    try:
                        json_file = json_files[0]
                        json_content = json.loads(json_file.read_text(encoding="utf-8"))

                        # 合并 JSON 页面数据
                        if "pages" in json_content:
                            has_json = True
                            page_offset = chunk_info.get("start_page", 1) - 1

                            for page in json_content["pages"]:
                                # 调整页码
                                if "page_number" in page:
                                    page["page_number"] += page_offset
                                json_pages.append(page)
                    except Exception as json_e:
                        logger.warning(f"⚠️  Failed to merge JSON for chunk {idx + 1}: {json_e}")

            # 保存合并后的 Markdown
            merged_md = "".join(markdown_parts)
            md_output = parent_output_dir / "result.md"
            md_output.write_text(merged_md, encoding="utf-8")
            logger.info(f"📄 Merged Markdown saved: {md_output}")

            # 保存合并后的 JSON (如果有)
            if has_json and json_pages:
                merged_json = {"pages": json_pages}
                json_output = parent_output_dir / "result.json"
                json_output.write_text(json.dumps(merged_json, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"📄 Merged JSON saved: {json_output}")

            # 规范化输出
            normalize_output(parent_output_dir)

            # 更新父任务状态
            self.task_db.update_task_status(
                task_id=parent_task_id, status="completed", result_path=str(parent_output_dir)
            )

            logger.info(f"✅ Parent task {parent_task_id} merged successfully")

            # 清理子任务的临时文件
            self._cleanup_child_task_files(children)

        except Exception as e:
            logger.error(f"❌ Failed to merge parent task {parent_task_id}: {e}")
            logger.exception(e)
            raise

    def _cleanup_child_task_files(self, children: list):
        """
        清理子任务的临时文件

        Args:
            children: 子任务列表
        """
        try:
            for child in children:
                # 删除子任务的分片 PDF 文件
                if child.get("file_path"):
                    chunk_file = Path(child["file_path"])
                    if chunk_file.exists() and chunk_file.is_file():
                        try:
                            chunk_file.unlink()
                            logger.debug(f"🗑️  Deleted chunk file: {chunk_file.name}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to delete chunk file {chunk_file.name}: {e}")

                # 可选: 删除子任务的结果目录 (如果需要节省空间)
                # 注意: 这会删除中间结果,可能影响调试
                # if child.get("result_path"):
                #     result_dir = Path(child["result_path"])
                #     if result_dir.exists() and result_dir.is_dir():
                #         try:
                #             shutil.rmtree(result_dir)
                #             logger.debug(f"🗑️  Deleted result dir: {result_dir.name}")
                #         except Exception as e:
                #             logger.warning(f"⚠️  Failed to delete result dir {result_dir.name}: {e}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup child task files: {e}")

    def _process_with_format_engine(self, file_path: str, options: dict, engine_name: Optional[str] = None) -> dict:
        """
        使用格式引擎处理专业领域格式文件

        Args:
            file_path: 文件路径
            options: 处理选项
            engine_name: 指定的引擎名称（如 fasta, genbank），为 None 时自动选择
        """
        # 获取语言设置
        lang = options.get("language", "en")

        # 根据指定的引擎名称或文件扩展名选择引擎
        if engine_name:
            # 用户明确指定了引擎
            engine = FormatEngineRegistry.get_engine(engine_name)
            if engine is None:
                raise ValueError(f"Format engine '{engine_name}' not found or not registered")

            # 验证文件是否适合该引擎
            if not engine.validate_file(file_path):
                raise ValueError(
                    f"File '{file_path}' is not supported by '{engine_name}' engine. "
                    f"Supported extensions: {', '.join(engine.SUPPORTED_EXTENSIONS)}"
                )

            # 使用指定引擎处理
            result = engine.parse(file_path, options={"language": lang})
        else:
            # 自动选择引擎（根据文件扩展名）
            engine = FormatEngineRegistry.get_engine_by_extension(file_path)
            if engine is None:
                raise ValueError(f"No format engine available for file: {file_path}")

            result = engine.parse(file_path, options={"language": lang})

        # 为每个任务创建专属输出目录（与其他引擎保持一致）
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果（与其他引擎保持一致的命名规范）
        # 主结果文件：result.md 和 result.json
        output_file = output_dir / "result.md"
        output_file.write_text(result["markdown"], encoding="utf-8")
        logger.info("📄 Main result saved: result.md")

        # 备份文件：使用原始文件名（便于调试）
        backup_md_file = output_dir / f"{Path(file_path).stem}_{result['format']}.md"
        backup_md_file.write_text(result["markdown"], encoding="utf-8")
        logger.info(f"📄 Backup saved: {backup_md_file.name}")

        # 也保存 JSON 结构化数据
        json_file = output_dir / "result.json"
        json_file.write_text(json.dumps(result["json_content"], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("📄 Main JSON saved: result.json")

        # 备份 JSON 文件
        backup_json_file = output_dir / f"{Path(file_path).stem}_{result['format']}.json"
        backup_json_file.write_text(json.dumps(result["json_content"], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"📄 Backup JSON saved: {backup_json_file.name}")

        # 规范化输出（统一文件名和目录结构）
        # Format Engine 已经输出标准格式，但仍然调用规范化器以确保一致性
        normalize_output(output_dir)

        return {
            "result_path": str(output_dir),  # 返回任务专属目录
            "content": result["content"],
            "json_path": str(json_file),
            "json_content": result["json_content"],
        }

    def decode_request(self, request):
        """
        解码请求

        LitServe 会调用这个方法来解析请求
        我们的请求格式: {"action": "health" | "poll"}
        """
        return request.get("action", "health")

    def predict(self, action):
        """
        处理请求

        Args:
            action: 请求动作
                - "health": 健康检查
                - "poll": 手动拉取任务（当 worker loop 禁用时）

        Returns:
            响应字典
        """
        if action == "health":
            # 健康检查
            vram_gb = None
            if "cuda" in str(self.device).lower():
                try:
                    vram_gb = get_vram(self.device.split(":")[-1])
                except Exception:
                    pass

            return {
                "status": "healthy",
                "worker_id": self.worker_id,
                "device": str(self.device),
                "vram_gb": vram_gb,
                "running": self.running,
                "current_task": self.current_task_id,
                "worker_loop_enabled": self.enable_worker_loop,
            }

        elif action == "poll":
            # 手动拉取任务（用于测试或禁用 worker loop 时）
            if self.enable_worker_loop:
                return {
                    "status": "skipped",
                    "message": "Worker is in auto-loop mode, manual polling is disabled",
                    "worker_id": self.worker_id,
                }

            task = self.task_db.pull_task()
            if task:
                task_id = task["task_id"]
                logger.info(f"📥 {self.worker_id} manually pulled task: {task_id}")

                try:
                    self._process_task(task)
                    logger.info(f"✅ {self.worker_id} completed task: {task_id}")

                    return {"status": "completed", "task_id": task["task_id"], "worker_id": self.worker_id}
                except Exception as e:
                    return {
                        "status": "failed",
                        "task_id": task["task_id"],
                        "error": str(e),
                        "worker_id": self.worker_id,
                    }
            else:
                # Worker 循环模式：返回状态信息
                return {
                    "status": "auto_mode",
                    "message": "Worker is running in auto-loop mode, tasks are processed automatically",
                    "worker_id": self.worker_id,
                    "worker_running": self.running,
                }

        else:
            return {
                "status": "error",
                "message": f'Invalid action: {action}. Use "health" or "poll".',
                "worker_id": self.worker_id,
            }

    def encode_response(self, response):
        """编码响应"""
        return response

    def teardown(self):
        """清理资源（Worker 关闭时调用）"""
        # 获取 worker_id（可能在 setup 失败时未初始化）
        worker_id = getattr(self, "worker_id", "unknown")

        logger.info(f"🛑 Worker {worker_id} shutting down...")

        # 设置 running 标志（如果已初始化）
        if hasattr(self, "running"):
            self.running = False

        # 等待 worker 线程结束
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

        logger.info(f"✅ Worker {worker_id} stopped")


def start_litserve_workers(
    output_dir=None,  # 默认从环境变量读取
    accelerator="auto",
    devices="auto",
    workers_per_device=1,
    port=8001,
    poll_interval=0.5,
    enable_worker_loop=True,
    paddleocr_vl_vllm_engine_enabled=False,
    paddleocr_vl_vllm_api_list=[],
    mineru_vllm_api_list=[],  # 新增参数
):
    """
    启动 LitServe Worker Pool

    Args:
        output_dir: 输出目录
        accelerator: 加速器类型 (auto/cuda/cpu/mps)
        devices: 使用的设备 (auto/[0,1,2])
        workers_per_device: 每个 GPU 的 worker 数量
        port: 服务端口
        poll_interval: Worker 拉取任务的间隔（秒）
        enable_worker_loop: 是否启用 worker 自动循环拉取任务
        paddleocr_vl_vllm_engine_enabled: 是否启用 PaddleOCR VL VLLM 引擎
        paddleocr_vl_vllm_api_list: PaddleOCR VL VLLM API 列表
        mineru_vllm_api_list: MinerU VLLM API 列表
    """

    def resolve_auto_accelerator():
        """
        当 accelerator 设置为 "auto" 时，使用元数据及环境信息自动检测最合适的加速器类型(不直接导入torch)

        Returns:
            str: 检测到的加速器类型 ("cuda" 或 "cpu")
        """
        try:
            from importlib.metadata import distribution

            distribution("torch")
            torch_is_installed = True
        except Exception as e:
            torch_is_installed = False
            logger.warning(f"Torch is not installed or cannot be imported: {e}")

        if torch_is_installed and check_cuda_with_nvidia_smi() > 0:
            return "cuda"
        return "cpu"

    # 如果没有指定输出目录，从环境变量读取
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        default_output = project_root / "data" / "output"
        output_dir = os.getenv("OUTPUT_PATH", str(default_output))

    logger.info("=" * 60)
    logger.info("🚀 Starting MinerU Tianshu LitServe Worker Pool")
    logger.info("=" * 60)
    logger.info(f"📂 Output Directory: {output_dir}")
    logger.info(f"💾 Devices: {devices}")
    logger.info(f"👷 Workers per Device: {workers_per_device}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🔄 Worker Loop: {'Enabled' if enable_worker_loop else 'Disabled'}")
    if enable_worker_loop:
        logger.info(f"⏱️  Poll Interval: {poll_interval}s")
    logger.info(f"🎮 Initial Accelerator setting: {accelerator}")

    if paddleocr_vl_vllm_engine_enabled:
        if not paddleocr_vl_vllm_api_list:
            logger.error(
                "请配置 --paddleocr-vl-vllm-api-list 参数，或移除 --paddleocr-vl-vllm-engine-enabled 以禁用 PaddleOCR VL VLLM 引擎"
            )
            sys.exit(1)
        logger.success(f"PaddleOCR VL VLLM 引擎已启用，API 列表为: {paddleocr_vl_vllm_api_list}")
    else:
        os.environ.pop("PADDLEOCR_VL_VLLM_ENABLED", None)
        logger.info("PaddleOCR VL VLLM 引擎已禁用")

    logger.info("=" * 60)

    # 1. 实例化 API 时传入数据
    api = MinerUWorkerAPI(
        output_dir=output_dir,
        poll_interval=poll_interval,
        enable_worker_loop=enable_worker_loop,
        paddleocr_vl_vllm_engine_enabled=paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=paddleocr_vl_vllm_api_list,
        mineru_vllm_api_list=mineru_vllm_api_list,  # ✅ 传入 MinerU API 列表
    )

    if accelerator == "auto":
        # 手动解析accelerator的具体设置
        accelerator = resolve_auto_accelerator()
        logger.info(f"💫 Auto-resolved Accelerator: {accelerator}")

    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=False,  # 不设置超时
    )

    # 注册优雅关闭处理器
    def graceful_shutdown(signum=None, frame=None):
        """处理关闭信号，优雅地停止 worker"""
        logger.info("🛑 Received shutdown signal, gracefully stopping workers...")
        # 注意：LitServe 会为每个设备创建多个 worker 实例
        # 这里的 api 只是模板，实际的 worker 实例由 LitServe 管理
        # teardown 会在每个 worker 进程中被调用
        if hasattr(api, "teardown"):
            api.teardown()
        sys.exit(0)

    # 注册信号处理器（Ctrl+C 等）
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # 注册 atexit 处理器（正常退出时调用）
    atexit.register(lambda: api.teardown() if hasattr(api, "teardown") else None)

    logger.info("✅ LitServe worker pool initialized")
    logger.info(f"📡 Listening on: http://0.0.0.0:{port}/predict")
    if enable_worker_loop:
        logger.info("🔁 Workers will continuously poll and process tasks")
    else:
        logger.info("🔄 Workers will wait for scheduler triggers")
    logger.info("=" * 60)

    # 启动服务器
    # 注意：LitServe 内置 MCP 已通过 monkeypatch 完全禁用（我们有独立的 MCP Server）
    server.run(port=port, generate_client_file=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MinerU Tianshu LitServe Worker Pool")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed files (default: from OUTPUT_PATH env or /app/output)",
    )
    parser.add_argument("--port", type=int, default=8001, help="Server port (default: 8001, or from WORKER_PORT env)")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Accelerator type (default: auto)",
    )
    parser.add_argument("--workers-per-device", type=int, default=1, help="Number of workers per device (default: 1)")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use, comma-separated (default: auto)")
    parser.add_argument(
        "--poll-interval", type=float, default=0.5, help="Worker poll interval in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--disable-worker-loop",
        action="store_true",
        help="Disable automatic worker loop (workers will wait for manual triggers)",
    )
    parser.add_argument(
        "--paddleocr-vl-vllm-engine-enabled",
        action="store_true",
        default=False,
        help="是否启用 PaddleOCR VL VLLM 引擎 (默认: False)",
    )
    parser.add_argument(
        "--paddleocr-vl-vllm-api-list",
        type=parse_list_arg,
        default=[],
        help='PaddleOCR VL VLLM API 列表（Python list 字面量格式，如: \'["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]\'）',
    )
    # 新增参数定义
    parser.add_argument(
        "--mineru-vllm-api-list",
        type=parse_list_arg,
        default=[],
        help='MinerU VLLM API 列表（Python list 字面量格式，如: \'["http://127.0.0.1:30024/v1"]\'）',
    )
    args = parser.parse_args()

    # ============================================================================
    # 从环境变量读取配置（如果命令行没有指定）
    # ============================================================================
    # 1. 如果没有通过命令行指定 devices，尝试自动检测或从环境变量读取
    devices = args.devices
    if devices == "auto":
        # 首先尝试从环境变量 CUDA_VISIBLE_DEVICES 读取（如果用户明确设置了）
        env_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if env_devices and env_devices.strip():
            devices = env_devices
            logger.info(f"📊 Using devices from CUDA_VISIBLE_DEVICES: {devices}")
        else:
            # 自动检测可用的 CUDA 设备
            try:
                import torch

                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    devices = ",".join(str(i) for i in range(device_count))
                    logger.info(f"📊 Auto-detected {device_count} CUDA devices: {devices}")
                else:
                    logger.info("📊 No CUDA devices available, using CPU mode")
                    devices = "auto"  # 保持 auto，让 LitServe 使用 CPU
            except Exception as e:
                logger.warning(f"⚠️  Failed to detect CUDA devices: {e}, using CPU mode")
                devices = "auto"

    # 2. 处理 devices 参数（支持逗号分隔的字符串）
    if devices != "auto":
        try:
            devices = [int(d.strip()) for d in devices.split(",")]
            logger.info(f"📊 Parsed devices: {devices}")
        except ValueError:
            logger.error(f"❌ Invalid devices format: {devices}. Use comma-separated integers (e.g., '0,1,2')")
            sys.exit(1)

    # 3. 如果没有通过命令行指定 workers-per-device，尝试从环境变量 WORKER_GPUS 读取
    workers_per_device = args.workers_per_device
    if args.workers_per_device == 1:  # 默认值
        env_workers = os.getenv("WORKER_GPUS")
        if env_workers:
            try:
                workers_per_device = int(env_workers)
                logger.info(f"📊 Using workers-per-device from WORKER_GPUS: {workers_per_device}")
            except ValueError:
                logger.warning(f"⚠️  Invalid WORKER_GPUS value: {env_workers}, using default: 1")

    # 4. 如果没有通过命令行指定 port，尝试从环境变量 WORKER_PORT 读取
    port = args.port
    if args.port == 8001:  # 默认值
        env_port = os.getenv("WORKER_PORT", "8001")
        try:
            port = int(env_port)
            logger.info(f"📊 Using port from WORKER_PORT env: {port}")
        except ValueError:
            logger.warning(f"⚠️  Invalid WORKER_PORT value: {env_port}, using default: 8001")
            port = 8001

    start_litserve_workers(
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        port=port,
        poll_interval=args.poll_interval,
        enable_worker_loop=not args.disable_worker_loop,
        paddleocr_vl_vllm_engine_enabled=args.paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=args.paddleocr_vl_vllm_api_list,
        mineru_vllm_api_list=args.mineru_vllm_api_list, # ✅ 传递参数
    )
