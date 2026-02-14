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
import shutil
from pathlib import Path
from typing import Optional
import multiprocessing
import requests

# Fix litserve MCP compatibility with mcp>=1.1.0
# Completely disable LitServe's internal MCP to avoid conflicts with our standalone MCP Server
import litserve as ls
from litserve.connector import check_cuda_with_nvidia_smi

try:
    from utils import parse_list_arg
except ImportError:
    # 兼容性回退
    def parse_list_arg(arg_str):
        import ast
        return ast.literal_eval(arg_str)

try:
    # Patch LitServe's MCP module to disable it completely
    import litserve.mcp as ls_mcp
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
    import warnings
    warnings.warn(f"Failed to patch litserve.mcp (MCP will be disabled): {e}")

from loguru import logger

# 添加父目录到路径以导入 MinerU
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from task_db import TaskDB
from output_normalizer import normalize_output
import importlib.util

# 尝试导入 markitdown
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    logger.warning("⚠️  markitdown not available, Office format parsing will be disabled")

# 检查 PaddleOCR-VL 是否可用
PADDLEOCR_VL_AVAILABLE = importlib.util.find_spec("paddleocr_vl") is not None
if PADDLEOCR_VL_AVAILABLE:
    logger.info("✅ PaddleOCR-VL engine available")
else:
    logger.info("ℹ️  PaddleOCR-VL not available (optional)")

# 检查 PaddleOCR-VL-VLLM 是否可用
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

# 检查水印去除引擎是否可用
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


# ==============================================================================
# VLLM Container Controller (互斥切换版 + 强力健康兜底修复)
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
            return docker.from_env()
        except Exception as e:
            logger.warning(f"⚠️  Docker client init failed: {e}")
            return None

    def ensure_service(self, target_container: str, conflict_container: str, health_url: str, timeout: int = 300):
        """
        确保目标容器运行，并关闭冲突容器 (互斥逻辑)
        """
        client = self._get_client()
        if client:
            try:
                # 1. 检查目标容器是否已经在运行
                try:
                    target = client.containers.get(target_container)
                    if target.status != 'running':
                        # 2. 停止冲突容器 (释放显存)
                        try:
                            conflict = client.containers.get(conflict_container)
                            if conflict.status == 'running':
                                logger.info(f"🛑 Stopping conflicting service {conflict_container} to free VRAM...")
                                conflict.stop()
                                logger.info(f"✅ Service {conflict_container} stopped.")
                        except Exception:
                            pass
                        
                        # 3. 启动目标容器
                        logger.info(f"🚀 Starting service {target_container} (Cold Start)...")
                        target.start()
                    else:
                        logger.info(f"✅ Target service {target_container} is already running.")
                except Exception as e:
                    logger.debug(f"❌ Container {target_container} not found or error: {e}")
            finally:
                try:
                    client.close()
                except:
                    pass
        else:
            logger.info(f"⚠️ Docker control skipped. Assuming {target_container} is externally managed.")

        # ✅ 核心修复：不管 docker api 有没有权限操作，也不管容器是不是已经在 running。
        # 只要配了 health_url，就强制轮询死锁等待，直到拿到 200 OK 才能放行执行任务！
        if health_url:
            self._wait_for_health(health_url, timeout)

    def _wait_for_health(self, url: str, timeout: int):
        """轮询健康检查接口，彻底解决 Connection Error"""
        start_time = time.time()
        logger.info(f"⏳ Waiting for vLLM models to load at {url} (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ vLLM Service is completely ready: {url} (took {int(time.time() - start_time)}s)")
                    return
            except Exception:
                # 屏蔽超时、拒绝连接等所有异常，默默重试
                pass
            
            time.sleep(3) # 每3秒探测一次
        
        raise TimeoutError(f"vLLM Service at {url} did not become ready in {timeout} seconds")


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
        self.mineru_vllm_api_list = mineru_vllm_api_list or []  
        
        ctx = multiprocessing.get_context("spawn")
        self._global_worker_counter = ctx.Value("i", 0)

        self.vllm_controller = VLLMController()

    def setup(self, device):
        """
        初始化 Worker (每个 GPU 上调用一次)
        """
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

        # 2. 分配 MinerU VLLM API 
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
        if "cuda:" in str(device):
            gpu_id = str(device).split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
            logger.info(f"🎯 [GPU Isolation] Set CUDA_VISIBLE_DEVICES={gpu_id} (Physical GPU {gpu_id} → Logical GPU 0)")
            logger.info("🎯 [GPU Isolation] Set MINERU_DEVICE_MODE=cuda:0")

        import socket

        # 配置模型下载源
        model_source = os.getenv("MODEL_DOWNLOAD_SOURCE", "auto").lower()

        if model_source in ["modelscope", "auto"]:
            try:
                import importlib.util
                if importlib.util.find_spec("modelscope") is not None:
                    logger.info("📦 Model download source: ModelScope (国内推荐)")
                else:
                    raise ImportError("modelscope not found")
            except ImportError:
                if model_source == "modelscope":
                    logger.warning("⚠️  ModelScope not available, falling back to HuggingFace")
                model_source = "huggingface"

        if model_source == "huggingface":
            hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
            logger.info(f"📦 Model download source: HuggingFace (via: {hf_endpoint})")
        elif model_source == "modelscope":
            os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
            logger.info("📦 Model download source: ModelScope")
        else:
            logger.warning(f"⚠️  Unknown model download source: {model_source}")

        self.device = device
        if "cuda" in str(device):
            self.accelerator = "cuda"
            self.engine_device = "cuda:0" 
        else:
            self.accelerator = "cpu"
            self.engine_device = "cpu" 

        logger.info(f"🎯 [Device] Accelerator: {self.accelerator}, Engine Device: {self.engine_device}")

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
                    vram = round(get_vram(device_mode))
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram)
                    logger.info(f"🎮 [MinerU VRAM] Detected: {vram}GB")
                except Exception as e:
                    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = "8"  
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

        # 初始化任务数据库
        db_path_env = os.getenv("DATABASE_PATH")
        if db_path_env:
            db_path = Path(db_path_env).resolve() 
            logger.info(f"📊 Using DATABASE_PATH from environment: {db_path_env} -> {db_path}")
        else:
            project_root = Path(__file__).parent.parent
            default_db = project_root / "data" / "db" / "mineru_tianshu.db"
            db_path = default_db.resolve()
            logger.warning(f"⚠️  DATABASE_PATH not set, using default: {db_path}")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path_str = str(db_path.absolute())
        logger.info(f"📊 Database path (absolute): {db_path_str}")

        self.task_db = TaskDB(db_path_str)

        try:
            stats = self.task_db.get_queue_stats()
            logger.info(f"📊 Database initialized: {db_path} (exists: {db_path.exists()})")
            logger.info(f"📊 TaskDB.db_path: {self.task_db.db_path}")
            logger.info(f"📊 Initial queue stats: {stats}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database or get stats: {e}")

        self.running = True
        self.current_task_id = None

        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"tianshu-{hostname}-{device}-{pid}"

        # 初始化可选的处理引擎
        self.markitdown = MarkItDown() if MARKITDOWN_AVAILABLE else None
        self.mineru_pipeline_engine = None 
        self.paddleocr_vl_engine = None 
        self.paddleocr_vl_vllm_engine = None 
        self.sensevoice_engine = None 
        self.video_engine = None 
        self.watermark_handler = None 

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

        logger.info("📦 Available Engines:")
        logger.info(f"   • MarkItDown: {'✅' if MARKITDOWN_AVAILABLE else '❌'}")
        logger.info(f"   • MinerU Pipeline: {'✅' if MINERU_PIPELINE_AVAILABLE else '❌'}")
        logger.info(f"   • PaddleOCR-VL: {'✅' if PADDLEOCR_VL_AVAILABLE else '❌'}")
        logger.info(f"   • SenseVoice: {'✅' if SENSEVOICE_AVAILABLE else '❌'}")
        logger.info(f"   • Video Engine: {'✅' if VIDEO_ENGINE_AVAILABLE else '❌'}")
        logger.info(f"   • Watermark Removal: {'✅' if WATERMARK_REMOVAL_AVAILABLE else '❌'}")
        logger.info(f"   • Format Engines: {'✅' if FORMAT_ENGINES_AVAILABLE else '❌'}")
        logger.info("")

        # 检测和初始化水印去除引擎
        if WATERMARK_REMOVAL_AVAILABLE and "cuda" in str(device).lower():
            try:
                logger.info("🎨 Initializing watermark removal engine...")
                from remove_watermark.pdf_watermark_handler import PDFWatermarkHandler
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

        if self.enable_worker_loop:
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"🔄 Worker loop started (poll_interval={self.poll_interval}s)")
        else:
            logger.info("⏸️  Worker loop disabled, waiting for manual triggers")

    def _worker_loop(self):
        """Worker 后台循环：持续拉取任务并处理"""
        logger.info(f"🔁 {self.worker_id} started task polling loop")

        try:
            stats = self.task_db.get_queue_stats()
            logger.info(f"📊 Initial queue stats: {stats}")
            logger.info(f"🗃️  Database path: {self.task_db.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to get initial queue stats: {e}")

        loop_count = 0
        last_stats_log = 0
        stats_log_interval = 60  # 每60次输出一次警告，避免刷屏

        while self.running:
            try:
                loop_count += 1
                task = self.task_db.get_next_task(worker_id=self.worker_id)

                if task:
                    task_id = task["task_id"]
                    self.current_task_id = task_id
                    logger.info(f"📥 {self.worker_id} pulled task: {task_id}")

                    try:
                        self._process_task(task)
                        logger.info(f"✅ {self.worker_id} completed task: {task_id}")
                    except Exception as e:
                        logger.error(f"❌ {self.worker_id} failed task {task_id}: {e}")
                        logger.exception(e)
                    finally:
                        self.current_task_id = None
                else:
                    if loop_count - last_stats_log >= stats_log_interval:
                        try:
                            stats = self.task_db.get_queue_stats()
                            pending = stats.get("pending", 0)
                            processing = stats.get("processing", 0)

                            if pending > 0:
                                logger.warning(
                                    f"⚠️  {self.worker_id} polling: {pending} pending tasks found but not pulled! "
                                    f"Processing: {processing}"
                                )
                            elif loop_count % 600 == 0: 
                                logger.info(f"💤 {self.worker_id} idle (loop #{loop_count}): No pending tasks.")
                        except Exception as e:
                            logger.error(f"❌ Failed to get queue stats: {e}")
                        last_stats_log = loop_count

                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"❌ Worker loop error (loop #{loop_count}): {e}")
                logger.exception(e)
                time.sleep(self.poll_interval)

    def _process_task(self, task: dict):
        """处理单个任务 (集成精准路由预判 + 互斥启动逻辑)"""
        task_id = task["task_id"]
        file_path = task["file_path"]
        options = json.loads(task.get("options", "{}"))
        parent_task_id = task.get("parent_task_id")
        backend = task.get("backend", "auto")
        
        try:
            file_ext = Path(file_path).suffix.lower()

            # ✅ 核心修复：提前准确预判实际处理所用的 backend，防止 auto 隐身漏掉 vLLM 互斥
            actual_backend = backend
            if actual_backend == "auto":
                if FORMAT_ENGINES_AVAILABLE and FormatEngineRegistry.is_supported(file_path):
                    actual_backend = "format"
                elif file_ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"] and SENSEVOICE_AVAILABLE:
                    actual_backend = "sensevoice"
                elif file_ext in [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"] and VIDEO_ENGINE_AVAILABLE:
                    actual_backend = "video"
                elif file_ext in [".pdf", ".png", ".jpg", ".jpeg"] and MINERU_PIPELINE_AVAILABLE:
                    actual_backend = "pipeline"
                elif file_ext in [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"]:
                    actual_backend = "pipeline" if options.get("convert_office_to_pdf") else "markitdown"
                elif self.markitdown:
                    actual_backend = "markitdown"

            # 1. 智能服务切换逻辑
            paddle_container = "tianshu-vllm-paddleocr"
            mineru_container = "tianshu-vllm-mineru"
            
            if actual_backend == "paddleocr-vl-vllm" and self.paddleocr_vl_vllm_api:
                base = self.paddleocr_vl_vllm_api.replace("/v1", "")
                health = f"{base}/health"
                self.vllm_controller.ensure_service(paddle_container, mineru_container, health)
                
            elif actual_backend in ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"] and self.mineru_vllm_api:
                base = self.mineru_vllm_api.replace("/v1", "")
                health = f"{base}/health"
                self.vllm_controller.ensure_service(mineru_container, paddle_container, health)

            # 【新增】Office 转 PDF 预处理
            office_extensions = [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"]
            if file_ext in office_extensions and options.get("convert_office_to_pdf", False):
                logger.info(f"📄 [Preprocessing] Converting Office to PDF: {file_path}")
                try:
                    pdf_path = self._convert_office_to_pdf(file_path)
                    original_file_path = file_path
                    file_path = pdf_path
                    file_ext = ".pdf"
                    logger.info(f"✅ [Preprocessing] Office converted, continuing with PDF: {pdf_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Office to PDF conversion failed: {e}")
                    logger.warning(f"   Falling back to MarkItDown for: {file_path}")

            # 检查是否需要拆分 PDF
            if file_ext == ".pdf" and not parent_task_id:
                if self._should_split_pdf(task_id, file_path, task, options):
                    return

            # 可选：预处理 - 去除水印
            if file_ext == ".pdf" and options.get("remove_watermark", False) and self.watermark_handler:
                logger.info(f"🎨 [Preprocessing] Removing watermark from PDF: {file_path}")
                try:
                    cleaned_pdf_path = self._preprocess_remove_watermark(file_path, options)
                    file_path = str(cleaned_pdf_path) 
                    logger.info(f"✅ [Preprocessing] Watermark removed, continuing with: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [Preprocessing] Watermark removal failed: {e}, continuing with original file")

            # 统一的引擎路由逻辑
            result = None 

            if actual_backend == "sensevoice":
                if not SENSEVOICE_AVAILABLE: raise ValueError("SenseVoice engine is not available")
                logger.info(f"🎤 Processing with SenseVoice: {file_path}")
                result = self._process_audio(file_path, options)

            elif actual_backend == "video":
                if not VIDEO_ENGINE_AVAILABLE: raise ValueError("Video processing engine is not available")
                logger.info(f"🎬 Processing with video engine: {file_path}")
                result = self._process_video(file_path, options)

            elif actual_backend == "paddleocr-vl":
                if not PADDLEOCR_VL_AVAILABLE: raise ValueError("PaddleOCR-VL engine is not available")
                logger.info(f"🔍 Processing with PaddleOCR-VL: {file_path}")
                result = self._process_with_paddleocr_vl(file_path, options)

            elif actual_backend == "paddleocr-vl-vllm":
                if not PADDLEOCR_VL_VLLM_AVAILABLE or not self.paddleocr_vl_vllm_engine_enabled or not self.paddleocr_vl_vllm_api_list:
                    raise ValueError("PaddleOCR-VL-VLLM engine is not available")
                logger.info(f"🔍 Processing with PaddleOCR-VL-VLLM: {file_path}")
                result = self._process_with_paddleocr_vl_vllm(file_path, options)
            
            elif actual_backend in ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]:
                if not MINERU_PIPELINE_AVAILABLE: raise ValueError(f"MinerU Pipeline engine is not available")
                logger.info(f"🔧 Processing with MinerU ({actual_backend}): {file_path}")
                options["parse_mode"] = actual_backend 
                result = self._process_with_mineru(file_path, options)

            elif actual_backend == "markitdown":
                if not self.markitdown: raise ValueError("MarkItDown engine is not available")
                logger.info(f"📄 Processing file with MarkItDown: {file_path}")
                result = self._process_with_markitdown(file_path)

            elif FORMAT_ENGINES_AVAILABLE and actual_backend == "format":
                logger.info(f"🧬 Processing with auto format engine: {file_path}")
                result = self._process_with_format_engine(file_path, options)

            elif FORMAT_ENGINES_AVAILABLE:
                engine = FormatEngineRegistry.get_engine(actual_backend)
                if engine is not None:
                    logger.info(f"🧬 Processing with format engine: {actual_backend}")
                    result = self._process_with_format_engine(file_path, options, engine_name=actual_backend)
                else:
                    raise ValueError(f"Unknown backend: {actual_backend}")
            else:
                raise ValueError(f"Unsupported file type/backend: file={file_path}, backend={actual_backend}")

            if result is None:
                raise ValueError(f"No result generated for backend: {actual_backend}, file: {file_path}")

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
                    logger.info(f"🔀 All subtasks completed, merging results for parent task {parent_id_to_merge}")
                    try:
                        self._merge_parent_task_results(parent_id_to_merge)
                    except Exception as merge_error:
                        logger.error(f"❌ Failed to merge parent task {parent_id_to_merge}: {merge_error}")
                        self.task_db.update_task_status(
                            parent_id_to_merge, "failed", error_message=f"Merge failed: {merge_error}"
                        )

            # 清理显存
            if "cuda" in str(self.device).lower():
                try:
                    from mineru.utils.model_utils import clean_memory
                    clean_memory()
                except Exception:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except: pass

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.task_db.update_task_status(task_id=task_id, status="failed", result_path=None, error_message=error_msg)
            if parent_task_id:
                self.task_db.on_child_task_failed(task_id, error_msg)
            raise

    def _process_with_mineru(self, file_path: str, options: dict) -> dict:
        """
        使用 MinerU 处理文档 + ✅ 终极路径扁平化
        彻底解决 /auto/ 或 TypeError 问题
        """
        if self.mineru_pipeline_engine is None:
            from mineru_pipeline import MinerUPipelineEngine
            self.mineru_pipeline_engine = MinerUPipelineEngine(
                device=self.engine_device,
                vlm_api_base=self.mineru_vllm_api 
            )
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ MinerU Pipeline engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ MinerU Pipeline engine loaded on CPU")

        # 将文件主名设为专属目标根目录
        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if "http-client" in options.get("parse_mode", "") and not options.get("server_url") and self.mineru_vllm_api:
            options["server_url"] = self.mineru_vllm_api.replace("/v1", "")

        result = self.mineru_pipeline_engine.parse(file_path, output_path=str(output_dir), options=options)

        # ✅ 核心修复：强制扁平化，不管 MinerU 在 result_path 里嵌套了多少层 auto
        actual_output_dir = Path(result.get("result_path", output_dir))
        
        if actual_output_dir.resolve() != output_dir.resolve():
            logger.info(f"🧹 Flattening deep nested dir from MinerU: {actual_output_dir} -> {output_dir}")
            if actual_output_dir.exists() and actual_output_dir.is_dir():
                # 把里面的东西掏出来放到根目录
                for item in actual_output_dir.iterdir():
                    dest = output_dir / item.name
                    if dest.exists():
                        if dest.is_dir(): shutil.rmtree(dest)
                        else: dest.unlink()
                    shutil.move(str(item), str(output_dir))
                
                # 回溯清理空的嵌套文件夹
                try:
                    curr = actual_output_dir
                    while curr.resolve() != output_dir.resolve() and curr.is_relative_to(output_dir):
                        if not any(curr.iterdir()):
                            curr.rmdir()
                        curr = curr.parent
                except Exception as e:
                    logger.debug(f"Cleanup empty dirs failed: {e}")

        # ✅ 修复 Bug 1：单参数安全调用规范化，去除报错的 target_dir
        normalize_output(output_dir)

        json_file = output_dir / "result.json"

        return {
            "result_path": str(output_dir), 
            "content": result.get("markdown", ""),
            "json_path": str(json_file) if json_file.exists() else None,
            "json_content": result.get("json_content"),
        }

    def _process_with_markitdown(self, file_path: str) -> dict:
        """使用 MarkItDown 处理 Office 文档（增强版：支持 DOCX 图片提取）"""
        if not self.markitdown:
            raise RuntimeError("MarkItDown is not available")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.markitdown.convert(file_path)
        markdown_content = result.text_content

        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".docx":
            try:
                from utils.docx_image_extractor import extract_images_from_docx, append_images_to_markdown
                images_dir = output_dir / "images"
                images = extract_images_from_docx(file_path, str(images_dir))
                if images:
                    markdown_content = append_images_to_markdown(markdown_content, images)
                    logger.info(f"🖼️  Extracted {len(images)} images from DOCX")
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract images from DOCX: {e}")

        output_file = output_dir / "result.md"
        output_file.write_text(markdown_content, encoding="utf-8")
        
        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": markdown_content}

    def _convert_office_to_pdf(self, file_path: str) -> str:
        """使用 LibreOffice 将 Office 文件转换为 PDF"""
        import subprocess
        import tempfile

        input_file = Path(file_path)
        final_pdf_file = input_file.parent / f"{input_file.stem}.pdf"

        if final_pdf_file.exists():
            final_pdf_file.unlink()

        logger.info(f"🔄 Converting Office to PDF: {input_file.name}")

        try:
            with tempfile.TemporaryDirectory(prefix="libreoffice_") as temp_dir:
                temp_dir_path = Path(temp_dir)
                temp_input = temp_dir_path / input_file.name
                shutil.copy2(input_file, temp_input)

                cmd = [
                    "libreoffice",
                    "--headless", 
                    "--convert-to", "pdf", 
                    "--outdir", str(temp_dir_path), 
                    str(temp_input), 
                ]
                result = subprocess.run(cmd, check=True, timeout=120, capture_output=True, text=True)

                temp_pdf = temp_dir_path / f"{input_file.stem}.pdf"
                if not temp_pdf.exists():
                    stderr_output = result.stderr if result.stderr else "No error output"
                    raise RuntimeError(f"LibreOffice conversion failed: output file not found: {temp_pdf}\nstderr: {stderr_output}")

                shutil.move(str(temp_pdf), str(final_pdf_file))
                logger.info(f"✅ Office converted to PDF: {final_pdf_file.name} ({final_pdf_file.stat().st_size / 1024:.1f} KB)")

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
        if self.accelerator == "cpu":
            raise RuntimeError("PaddleOCR-VL requires GPU and is not supported in CPU mode.")

        if self.paddleocr_vl_engine is None:
            from paddleocr_vl import PaddleOCRVLEngine
            self.paddleocr_vl_engine = PaddleOCRVLEngine(device="cuda:0", model_name="PaddleOCR-VL-1.5")
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"✅ PaddleOCR-VL engine loaded on cuda:0 (physical GPU {gpu_id})")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.paddleocr_vl_engine.parse(file_path, output_path=str(output_dir))
        
        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_with_paddleocr_vl_vllm(self, file_path: str, options: dict) -> dict:
        """使用 PaddleOCR-VL VLLM 处理图片或 PDF"""
        if self.accelerator == "cpu":
            raise RuntimeError("PaddleOCR-VL VLLM requires GPU and is not supported in CPU mode.")

        if self.paddleocr_vl_vllm_engine is None:
            from paddleocr_vl_vllm import PaddleOCRVLVLLMEngine
            self.paddleocr_vl_vllm_engine = PaddleOCRVLVLLMEngine(
                device="cuda:0", 
                vllm_api_base=self.paddleocr_vl_vllm_api,
                model_name="PaddleOCR-VL-1.5-0.9B"
            )
            gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            logger.info(f"✅ PaddleOCR-VL VLLM engine loaded on cuda:0 (physical GPU {gpu_id})")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.paddleocr_vl_vllm_engine.parse(file_path, output_path=str(output_dir))

        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_audio(self, file_path: str, options: dict) -> dict:
        """使用 SenseVoice 处理音频文件"""
        if self.sensevoice_engine is None:
            from audio_engines import SenseVoiceEngine
            self.sensevoice_engine = SenseVoiceEngine(device=self.engine_device)
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ SenseVoice engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ SenseVoice engine loaded on CPU")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.sensevoice_engine.parse(
            audio_path=file_path,
            output_path=str(output_dir),
            language=options.get("lang", "auto"),
            use_itn=options.get("use_itn", True),
            enable_speaker_diarization=options.get("enable_speaker_diarization", False),
        )

        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": result.get("markdown", "")}

    def _process_video(self, file_path: str, options: dict) -> dict:
        """使用视频处理引擎处理视频文件"""
        if self.video_engine is None:
            from video_engines import VideoProcessingEngine
            self.video_engine = VideoProcessingEngine(device=self.engine_device)
            if self.accelerator == "cuda":
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                logger.info(f"✅ Video processing engine loaded on cuda:0 (physical GPU {gpu_id})")
            else:
                logger.info("✅ Video processing engine loaded on CPU")

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

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

        output_file = output_dir / f"{Path(file_path).stem}_video_analysis.md"
        output_file.write_text(result["markdown"], encoding="utf-8")
        
        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {"result_path": str(output_dir), "content": result["markdown"]}

    def _preprocess_remove_watermark(self, file_path: str, options: dict) -> Path:
        """预处理：去除 PDF 水印"""
        if not self.watermark_handler:
            raise RuntimeError("Watermark removal is not available (CUDA required)")

        output_file = Path(self.output_dir) / f"{Path(file_path).stem}_no_watermark.pdf"

        kwargs = {}
        if "auto_detect" in options: kwargs["auto_detect"] = options["auto_detect"]
        if "force_scanned" in options: kwargs["force_scanned"] = options["force_scanned"]
        if "remove_text" in options: kwargs["remove_text"] = options["remove_text"]
        if "remove_images" in options: kwargs["remove_images"] = options["remove_images"]
        if "remove_annotations" in options: kwargs["remove_annotations"] = options["remove_annotations"]
        if "watermark_keywords" in options: kwargs["keywords"] = options["watermark_keywords"]
        if "watermark_dpi" in options: kwargs["dpi"] = options["watermark_dpi"]
        if "watermark_conf_threshold" in options: kwargs["conf_threshold"] = options["watermark_conf_threshold"]
        if "watermark_dilation" in options: kwargs["dilation"] = options["watermark_dilation"]

        cleaned_pdf_path = self.watermark_handler.remove_watermark(
            input_path=file_path, output_path=str(output_file), **kwargs
        )
        return cleaned_pdf_path

    def _should_split_pdf(self, task_id: str, file_path: str, task: dict, options: dict) -> bool:
        """判断 PDF 是否需要拆分，如果需要则执行拆分"""
        from utils.pdf_utils import get_pdf_page_count, split_pdf_file

        if os.getenv("PDF_SPLIT_ENABLED", "true").lower() != "true":
            return False

        pdf_split_threshold = int(os.getenv("PDF_SPLIT_THRESHOLD_PAGES", "500"))
        pdf_split_chunk_size = int(os.getenv("PDF_SPLIT_CHUNK_SIZE", "500"))

        try:
            page_count = get_pdf_page_count(Path(file_path))
            logger.info(f"📄 PDF has {page_count} pages (threshold: {pdf_split_threshold})")

            if page_count <= pdf_split_threshold:
                return False

            logger.info(f"🔀 Large PDF detected ({page_count} pages), splitting into chunks of {pdf_split_chunk_size} pages")
            
            self.task_db.convert_to_parent_task(task_id, child_count=0)
            split_dir = Path(self.output_dir) / "splits" / task_id
            split_dir.mkdir(parents=True, exist_ok=True)

            chunks = split_pdf_file(
                pdf_path=Path(file_path),
                output_dir=split_dir,
                chunk_size=pdf_split_chunk_size,
                parent_task_id=task_id,
            )

            logger.info(f"✂️  PDF split into {len(chunks)} chunks")

            backend = task.get("backend", "auto")
            priority = task.get("priority", 0)
            user_id = task.get("user_id")

            for chunk_info in chunks:
                chunk_options = options.copy()
                chunk_options["chunk_info"] = {
                    "start_page": chunk_info["start_page"],
                    "end_page": chunk_info["end_page"],
                    "page_count": chunk_info["page_count"],
                }

                child_task_id = self.task_db.create_child_task(
                    parent_task_id=task_id,
                    file_name=f"{Path(file_path).stem}_pages_{chunk_info['start_page']}-{chunk_info['end_page']}.pdf",
                    file_path=chunk_info["path"],
                    backend=backend,
                    options=chunk_options,
                    priority=priority,
                    user_id=user_id,
                )
                logger.info(f"  ✅ Created subtask {child_task_id}: pages {chunk_info['start_page']}-{chunk_info['end_page']}")

            self.task_db.convert_to_parent_task(task_id, child_count=len(chunks))
            logger.info(f"🎉 Large PDF split complete: {len(chunks)} subtasks created for parent task {task_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to split PDF: {e}")
            logger.warning("⚠️  Falling back to processing as single task")
            return False

    def _merge_parent_task_results(self, parent_task_id: str):
        """合并父任务的所有子任务结果"""
        try:
            parent_task = self.task_db.get_task_with_children(parent_task_id)
            if not parent_task:
                raise ValueError(f"Parent task {parent_task_id} not found")

            children = parent_task.get("children", [])
            if not children:
                raise ValueError(f"No child tasks found for parent {parent_task_id}")

            children.sort(key=lambda x: json.loads(x.get("options", "{}")).get("chunk_info", {}).get("start_page", 0))
            logger.info(f"🔀 Merging {len(children)} subtask results for parent task {parent_task_id}")

            parent_output_dir = Path(self.output_dir) / Path(parent_task["file_path"]).stem
            parent_output_dir.mkdir(parents=True, exist_ok=True)

            markdown_parts = []
            json_pages = []
            has_json = False

            for idx, child in enumerate(children):
                if child["status"] != "completed":
                    logger.warning(f"⚠️  Child task {child['task_id']} not completed (status: {child['status']})")
                    continue

                result_dir = Path(child["result_path"])
                chunk_info = json.loads(child.get("options", "{}")).get("chunk_info", {})

                md_files = list(result_dir.rglob("*.md"))
                if md_files:
                    md_file = next((f for f in md_files if f.name == "result.md"), md_files[0])
                    content = md_file.read_text(encoding="utf-8")
                    if chunk_info: markdown_parts.append(f"\n\n\n\n")
                    markdown_parts.append(content)
                    logger.info(f"   ✅ Merged chunk {idx + 1}/{len(children)}: pages {chunk_info.get('start_page', '?')}-{chunk_info.get('end_page', '?')}")

                json_files = [f for f in result_dir.rglob("*.json") if f.name in ["content.json", "result.json"] or "_content_list.json" in f.name]
                if json_files:
                    try:
                        json_file = json_files[0]
                        json_content = json.loads(json_file.read_text(encoding="utf-8"))
                        if "pages" in json_content:
                            has_json = True
                            page_offset = chunk_info.get("start_page", 1) - 1
                            for page in json_content["pages"]:
                                if "page_number" in page: page["page_number"] += page_offset
                                json_pages.append(page)
                    except Exception as json_e:
                        logger.warning(f"⚠️  Failed to merge JSON for chunk {idx + 1}: {json_e}")

            merged_md = "".join(markdown_parts)
            md_output = parent_output_dir / "result.md"
            md_output.write_text(merged_md, encoding="utf-8")
            logger.info(f"📄 Merged Markdown saved: {md_output}")

            if has_json and json_pages:
                merged_json = {"pages": json_pages}
                json_output = parent_output_dir / "result.json"
                json_output.write_text(json.dumps(merged_json, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"📄 Merged JSON saved: {json_output}")

            # ✅ 修复 Bug 1：单参数安全调用
            normalize_output(parent_output_dir)

            self.task_db.update_task_status(parent_task_id, status="completed", result_path=str(parent_output_dir))
            logger.info(f"✅ Parent task {parent_task_id} merged successfully")

            self._cleanup_child_task_files(children)

        except Exception as e:
            logger.error(f"❌ Failed to merge parent task {parent_task_id}: {e}")
            logger.exception(e)
            raise

    def _cleanup_child_task_files(self, children: list):
        """清理子任务的临时文件"""
        try:
            for child in children:
                if child.get("file_path"):
                    chunk_file = Path(child["file_path"])
                    if chunk_file.exists() and chunk_file.is_file():
                        try:
                            chunk_file.unlink()
                            logger.debug(f"🗑️  Deleted chunk file: {chunk_file.name}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to delete chunk file {chunk_file.name}: {e}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup child task files: {e}")

    def _process_with_format_engine(self, file_path: str, options: dict, engine_name: Optional[str] = None) -> dict:
        """使用格式引擎处理专业领域格式文件"""
        lang = options.get("language", "en")
        
        if engine_name:
            engine = FormatEngineRegistry.get_engine(engine_name)
            if engine is None:
                raise ValueError(f"Format engine '{engine_name}' not found or not registered")
            if not engine.validate_file(file_path):
                raise ValueError(f"File '{file_path}' is not supported by '{engine_name}' engine.")
            result = engine.parse(file_path, options={"language": lang})
        else:
            engine = FormatEngineRegistry.get_engine_by_extension(file_path)
            if engine is None:
                raise ValueError(f"No format engine available for file: {file_path}")
            result = engine.parse(file_path, options={"language": lang})

        output_dir = Path(self.output_dir) / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "result.md"
        output_file.write_text(result["markdown"], encoding="utf-8")
        logger.info("📄 Main result saved: result.md")

        backup_md_file = output_dir / f"{Path(file_path).stem}_{result['format']}.md"
        backup_md_file.write_text(result["markdown"], encoding="utf-8")
        logger.info(f"📄 Backup saved: {backup_md_file.name}")

        json_file = output_dir / "result.json"
        json_file.write_text(json.dumps(result["json_content"], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("📄 Main JSON saved: result.json")

        backup_json_file = output_dir / f"{Path(file_path).stem}_{result['format']}.json"
        backup_json_file.write_text(json.dumps(result["json_content"], indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"📄 Backup JSON saved: {backup_json_file.name}")

        # ✅ 修复 Bug 1：单参数安全调用
        normalize_output(output_dir)

        return {
            "result_path": str(output_dir),  
            "content": result["content"],
            "json_path": str(json_file),
            "json_content": result["json_content"],
        }

    def decode_request(self, request):
        return request.get("action", "health")

    def encode_response(self, response):
        return response

    def predict(self, action):
        if action == "health":
            vram_gb = None
            if "cuda" in str(self.device).lower():
                try:
                    from mineru.utils.model_utils import get_vram
                    vram_gb = round(get_vram(self.device.split(":")[-1]))
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
                    return {"status": "completed", "task_id": task_id, "worker_id": self.worker_id}
                except Exception as e:
                    return {
                        "status": "failed",
                        "task_id": task_id,
                        "error": str(e),
                        "worker_id": self.worker_id,
                    }
            else:
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

    def teardown(self):
        worker_id = getattr(self, "worker_id", "unknown")
        logger.info(f"🛑 Worker {worker_id} shutting down...")
        self.running = False
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info(f"✅ Worker {worker_id} stopped")


def start_litserve_workers(
    output_dir=None,
    accelerator="auto",
    devices="auto",
    workers_per_device=1,
    port=8001,
    poll_interval=0.5,
    enable_worker_loop=True,
    paddleocr_vl_vllm_engine_enabled=False,
    paddleocr_vl_vllm_api_list=[],
    mineru_vllm_api_list=[],
):
    def resolve_auto_accelerator():
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
        logger.info(f"⏱️  Poll Interval: {poll_interval}s")
    logger.info(f"🎮 Initial Accelerator setting: {accelerator}")

    if paddleocr_vl_vllm_engine_enabled:
        if not paddleocr_vl_vllm_api_list:
            logger.error("请配置 --paddleocr-vl-vllm-api-list 参数，或移除 --paddleocr-vl-vllm-engine-enabled 以禁用 PaddleOCR VL VLLM 引擎")
            sys.exit(1)
        logger.success(f"PaddleOCR VL VLLM 引擎已启用，API 列表为: {paddleocr_vl_vllm_api_list}")
    else:
        os.environ.pop("PADDLEOCR_VL_VLLM_ENABLED", None)
        logger.info("PaddleOCR VL VLLM 引擎已禁用")

    logger.info("=" * 60)

    api = MinerUWorkerAPI(
        output_dir=output_dir,
        poll_interval=poll_interval,
        enable_worker_loop=enable_worker_loop,
        paddleocr_vl_vllm_engine_enabled=paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=paddleocr_vl_vllm_api_list,
        mineru_vllm_api_list=mineru_vllm_api_list, 
    )

    if accelerator == "auto":
        accelerator = resolve_auto_accelerator()
        logger.info(f"💫 Auto-resolved Accelerator: {accelerator}")

    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=False,
    )

    def graceful_shutdown(signum=None, frame=None):
        logger.info("🛑 Received shutdown signal, gracefully stopping workers...")
        if hasattr(api, "teardown"):
            api.teardown()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    atexit.register(lambda: api.teardown() if hasattr(api, "teardown") else None)

    logger.info("✅ LitServe worker pool initialized")
    logger.info(f"📡 Listening on: http://0.0.0.0:{port}/predict")
    if enable_worker_loop:
        logger.info("🔁 Workers will continuously poll and process tasks")
    else:
        logger.info("🔄 Workers will wait for scheduler triggers")
    logger.info("=" * 60)

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
    parser.add_argument(
        "--mineru-vllm-api-list",
        type=parse_list_arg,
        default=[],
        help='MinerU VLLM API 列表（Python list 字面量格式，如: \'["http://127.0.0.1:30024/v1"]\'）',
    )
    args = parser.parse_args()

    # 从环境变量读取配置
    devices = args.devices
    if devices == "auto":
        env_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if env_devices and env_devices.strip():
            devices = env_devices
            logger.info(f"📊 Using devices from CUDA_VISIBLE_DEVICES: {devices}")
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    devices = ",".join(str(i) for i in range(device_count))
                    logger.info(f"📊 Auto-detected {device_count} CUDA devices: {devices}")
                else:
                    logger.info("📊 No CUDA devices available, using CPU mode")
                    devices = "auto" 
            except Exception as e:
                logger.warning(f"⚠️  Failed to detect CUDA devices: {e}, using CPU mode")
                devices = "auto"

    if devices != "auto":
        try:
            devices = [int(d.strip()) for d in devices.split(",")]
            logger.info(f"📊 Parsed devices: {devices}")
        except ValueError:
            logger.error(f"❌ Invalid devices format: {devices}. Use comma-separated integers (e.g., '0,1,2')")
            sys.exit(1)

    workers_per_device = args.workers_per_device
    if args.workers_per_device == 1: 
        env_workers = os.getenv("WORKER_GPUS")
        if env_workers:
            try:
                workers_per_device = int(env_workers)
                logger.info(f"📊 Using workers-per-device from WORKER_GPUS: {workers_per_device}")
            except ValueError:
                logger.warning(f"⚠️  Invalid WORKER_GPUS value: {env_workers}, using default: 1")

    port = args.port
    if args.port == 8001: 
        env_port = os.getenv("WORKER_PORT", "8001")
        try:
            port = int(env_port)
            logger.info(f"📊 Using port from WORKER_PORT env: {port}")
        except ValueError:
            logger.warning(f"⚠️  Invalid WORKER_PORT value: {env_port}, using default: 8001")
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
        mineru_vllm_api_list=args.mineru_vllm_api_list,
    )
