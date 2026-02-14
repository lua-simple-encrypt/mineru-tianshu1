"""
MinerU Pipeline Engine
单例模式，每个进程只加载一次模型
使用 MinerU 处理 PDF 和图片
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import img2pdf


class MinerUPipelineEngine:
    """
    MinerU Pipeline 引擎

    特性：
    - 单例模式
    - 封装 MinerU 的 do_parse 调用
    - 支持 pipeline, vlm-auto-engine, hybrid-auto-engine 模式
    - 支持 VLLM API 调用 (自动切换到 http-client 模式)
    - 支持丰富的输出选项配置
    """

    _instance: Optional["MinerUPipelineEngine"] = None
    _lock = Lock()
    _pipeline = None  # 这里的 pipeline 实际上是 do_parse 函数
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0", vlm_api_base: str = None):
        """
        初始化引擎

        Args:
            device: 设备 (cuda:0, cuda:1 等)
            vlm_api_base: VLLM API 地址 (例如 http://vllm-mineru:30024/v1)
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.vlm_api_base = vlm_api_base  # 保存 VLLM API 地址

            # 从 device 字符串中提取 GPU ID
            if "cuda:" in device:
                self.gpu_id = device.split(":")[-1]
            else:
                self.gpu_id = "0"

            self._initialized = True
            logger.info(f"🔧 MinerU Pipeline Engine initialized on {device}")
            if self.vlm_api_base:
                logger.info(f"   VLLM API Base: {self.vlm_api_base}")

    def _load_pipeline(self):
        """延迟加载 MinerU 管道 (do_parse)"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("=" * 60)
            logger.info("📥 Loading MinerU Pipeline (do_parse)...")
            logger.info("=" * 60)

            try:
                # 延迟导入 do_parse，避免过早初始化模型
                from mineru.cli.common import do_parse

                self._pipeline = do_parse

                logger.info("=" * 60)
                logger.info("✅ MinerU Pipeline loaded successfully!")
                logger.info("=" * 60)

                return self._pipeline

            except ImportError:
                logger.error("❌ Failed to import mineru.cli.common.do_parse")
                raise
            except Exception as e:
                logger.error(f"❌ Error loading MinerU pipeline: {e}")
                raise

    def cleanup(self):
        """清理显存"""
        try:
            from mineru.utils.model_utils import clean_memory

            clean_memory()
            logger.debug("🧹 MinerU: Memory cleanup completed")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")

    def parse(self, file_path: str, output_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理文件

        Args:
            file_path: 输入文件路径
            output_path: 输出目录路径
            options: 处理选项

        Returns:
            包含结果的字典
        """
        options = options or {}
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_stem = Path(file_path).stem
        file_ext = Path(file_path).suffix.lower()

        # 1. 确定 Backend (处理模式) 和 Server URL
        # options["parse_mode"] 来自前端 API: pipeline | vlm-auto-engine | hybrid-auto-engine | vlm-http-client | hybrid-http-client
        user_backend = options.get("parse_mode", "pipeline")
        if user_backend == "auto":
            user_backend = "pipeline"

        backend = user_backend
        server_url = options.get("server_url")  # 优先使用 options 中的 server_url (Client 模式)

        # 智能切换：如果配置了本地 vlm_api_base 且没指定 server_url，尝试自动使用本地服务加速
        if not server_url and self.vlm_api_base:
            if user_backend == "vlm-auto-engine":
                backend = "vlm-http-client"
                # 去掉 /v1 后缀，因为 MinerU 客户端通常只需要 base url
                server_url = self.vlm_api_base.replace("/v1", "")
                logger.info(f"🔄 [Accelerate] Switching backend to {backend} using local vLLM: {server_url}")
            elif user_backend == "hybrid-auto-engine":
                backend = "hybrid-http-client"
                server_url = self.vlm_api_base.replace("/v1", "")
                logger.info(f"🔄 [Accelerate] Switching backend to {backend} using local vLLM: {server_url}")
        
        # 记录非 Client 模式的情况
        if backend in ["vlm-auto-engine", "hybrid-auto-engine"] and not server_url:
            logger.info(f"ℹ️  Running {backend} locally (No vLLM configured)")

        # 2. 确定 Method (解析方法)
        # options["method"] 来自 API: auto | txt | ocr
        parse_method = options.get("method", "auto")
        # 兼容旧参数 force_ocr
        if options.get("force_ocr"):
            parse_method = "ocr"

        # 3. 提取其他高级选项 (从 options 中获取，如果没有则使用默认值)
        
        # 内容识别
        formula_enable = options.get("formula_enable", True)
        table_enable = options.get("table_enable", True)
        
        # 输出控制 (默认开启所有调试输出，方便用户下载)
        f_draw_layout_bbox = options.get("draw_layout_bbox", True)      
        f_draw_span_bbox = options.get("draw_span_bbox", True)          
        f_dump_md = options.get("dump_markdown", True)                  
        f_dump_middle_json = options.get("dump_middle_json", True)      
        f_dump_model_output = options.get("dump_model_output", True)    
        f_dump_content_list = options.get("dump_content_list", True)    
        f_dump_orig_pdf = options.get("dump_orig_pdf", True)            

        # 兼容旧参数
        if "draw_layout" in options:
            f_draw_layout_bbox = options["draw_layout"]
        if "draw_span" in options:
            f_draw_span_bbox = options["draw_span"]
        
        # 页面范围
        start_page_id = options.get("start_page_id", options.get("start_page", 0))
        end_page_id = options.get("end_page_id", options.get("end_page", None))

        # 处理无效值
        if start_page_id is None or str(start_page_id).strip() == "":
            start_page_id = 0
        else:
            start_page_id = int(start_page_id)

        if end_page_id is not None:
             if end_page_id == -1 or str(end_page_id).strip() == "":
                 end_page_id = None
             else:
                 end_page_id = int(end_page_id)

        logger.info(f"🚀 MinerU Engine starting")
        logger.info(f"   Backend: {backend}")
        logger.info(f"   Method: {parse_method}")
        logger.info(f"   Page Range: {start_page_id} -> {end_page_id if end_page_id is not None else 'End'}")
        if server_url:
            logger.info(f"   Server URL: {server_url}")

        # 加载管道 (do_parse 函数)
        do_parse_func = self._load_pipeline()

        try:
            # 读取文件为字节
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # MinerU 的 do_parse 只支持 PDF 格式
            # 图片文件需要先转换为 PDF
            if file_ext in [".png", ".jpg", ".jpeg"]:
                logger.info("🖼️  Converting image to PDF for MinerU processing...")
                try:
                    pdf_bytes = img2pdf.convert(file_bytes)
                    file_name = f"{file_stem}.pdf"  # 使用 .pdf 扩展名
                    logger.info(f"✅ Image converted: {file_name} ({len(pdf_bytes)} bytes)")
                except Exception as e:
                    logger.error(f"❌ Image conversion failed: {e}")
                    raise ValueError(f"Failed to convert image to PDF: {e}")
            else:
                # PDF 文件直接使用
                pdf_bytes = file_bytes
                file_name = Path(file_path).name

            # 获取语言设置
            lang = options.get("lang", "auto")
            if lang == "auto":
                lang = "ch"  # 默认中文/通用
            logger.info(f"🌐 Language set to '{lang}'")

            # 调用 MinerU (do_parse)
            # 严格按照 do_parse 函数签名传参
            do_parse_func(
                output_dir=str(output_dir),            # 输出目录
                pdf_file_names=[file_name],            # 文件名列表
                pdf_bytes_list=[pdf_bytes],            # 文件字节列表
                p_lang_list=[lang],                    # 语言列表
                
                # 核心控制参数
                backend=backend,                       # 后端
                parse_method=parse_method,             # 解析方法
                server_url=server_url,                 # VLLM 地址
                
                # 功能开关
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                formula_enable=formula_enable,
                table_enable=table_enable,
                
                # 输出控制
                f_draw_layout_bbox=f_draw_layout_bbox,
                f_draw_span_bbox=f_draw_span_bbox,
                f_dump_md=f_dump_md,
                f_dump_middle_json=f_dump_middle_json,
                f_dump_model_output=f_dump_model_output,
                f_dump_orig_pdf=f_dump_orig_pdf,
                f_dump_content_list=f_dump_content_list
            )

            # MinerU 新版输出结构: {output_dir}/{file_name}/auto/{file_stem}.md
            # 递归查找 markdown 文件
            md_files = list(output_dir.rglob("*.md"))

            if md_files:
                # 使用第一个找到的 md 文件
                md_file = md_files[0]
                logger.info(f"✅ Found MinerU output: {md_file}")
                content = md_file.read_text(encoding="utf-8")

                # 返回实际的输出目录（包含 auto/ 子目录）
                actual_output_dir = md_file.parent

                # 查找 JSON 文件
                # MinerU 输出格式: {filename}_content_list.json
                json_files = [
                    f
                    for f in actual_output_dir.rglob("*.json")
                    if "_content_list.json" in f.name and not f.parent.name.startswith("page_")
                ]

                result = {
                    "markdown": content,
                    "result_path": str(actual_output_dir),  # 返回包含所有输出的目录
                }

                # 如果找到 JSON 文件，也读取它
                if json_files:
                    json_file = json_files[0]
                    logger.info(f"✅ Found MinerU JSON output: {json_file}")
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            json_content = json.load(f)
                        result["json_path"] = str(json_file)
                        result["json_content"] = json_content
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load JSON: {e}")
                else:
                    logger.info("ℹ️  No JSON output found (MinerU may not generate it by default)")

                return result
            else:
                # 如果找不到 md 文件，列出输出目录内容以便调试
                logger.error("❌ MinerU output directory structure:")
                for item in output_dir.rglob("*"):
                    logger.error(f"   {item}")
                raise FileNotFoundError(f"MinerU output not found in: {output_dir}")

        finally:
            self.cleanup()


# 全局单例
_engine = None


def get_engine(vlm_api_base: str = None) -> MinerUPipelineEngine:
    """
    获取全局引擎实例
    
    Args:
        vlm_api_base: 可选，VLLM API 地址。如果单例已存在，此参数将被忽略。
    """
    global _engine
    if _engine is None:
        _engine = MinerUPipelineEngine(vlm_api_base=vlm_api_base)
    return _engine
