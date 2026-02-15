"""
MinerU Pipeline Engine
单例模式，每个进程只加载一次模型
使用 MinerU 处理 PDF 和图片

修复说明：
- 强制使用安全文件名 'result.pdf' 进行内部处理，解决中文文件名导致的路径问题
- 增加 .json 结果查找作为 .md 缺失时的降级方案
- 增加 layout.pdf 存在的容错处理
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
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.vlm_api_base = vlm_api_base

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
                from mineru.cli.common import do_parse
                self._pipeline = do_parse
                logger.info("✅ MinerU Pipeline loaded successfully!")
                return self._pipeline
            except Exception as e:
                logger.error(f"❌ Error loading MinerU pipeline: {e}")
                raise

    def cleanup(self):
        """清理显存"""
        try:
            from mineru.utils.model_utils import clean_memory
            clean_memory()
            logger.debug("🧹 MinerU: Memory cleanup completed")
        except Exception:
            pass

    def parse(self, file_path: str, output_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理文件
        """
        options = options or {}
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()

        # =========================================================================
        # 1. 确定 Backend
        # =========================================================================
        user_backend = options.get("parse_mode", "pipeline")
        if user_backend == "auto":
            user_backend = "pipeline"

        backend = user_backend
        server_url = options.get("server_url")

        # 智能切换 VLLM
        if not server_url and self.vlm_api_base:
            if user_backend == "vlm-auto-engine":
                backend = "vlm-http-client"
                server_url = self.vlm_api_base.replace("/v1", "")
                logger.info(f"🔄 [Accelerate] Switching to {backend} using local vLLM")
            elif user_backend == "hybrid-auto-engine":
                backend = "hybrid-http-client"
                server_url = self.vlm_api_base.replace("/v1", "")
                logger.info(f"🔄 [Accelerate] Switching to {backend} using local vLLM")

        # =========================================================================
        # 2. 准备参数
        # =========================================================================
        parse_method = options.get("method", "auto")
        if options.get("force_ocr"):
            parse_method = "ocr"

        # 功能开关
        formula_enable = options.get("formula_enable", True)
        table_enable = options.get("table_enable", True)
        
        # 输出控制
        f_draw_layout_bbox = options.get("draw_layout_bbox", True)      
        f_draw_span_bbox = options.get("draw_span_bbox", True)          
        f_dump_md = options.get("dump_markdown", True)                  
        f_dump_middle_json = options.get("dump_middle_json", True)      
        f_dump_model_output = options.get("dump_model_output", True)    
        f_dump_content_list = options.get("dump_content_list", True)    
        f_dump_orig_pdf = options.get("dump_orig_pdf", True)            

        # 页面范围
        start_page_id = options.get("start_page_id", options.get("start_page", 0))
        end_page_id = options.get("end_page_id", options.get("end_page", None))
        
        if start_page_id is None: start_page_id = 0
        else: start_page_id = int(start_page_id)
        
        if end_page_id == -1 or str(end_page_id).strip() == "": end_page_id = None
        elif end_page_id is not None: end_page_id = int(end_page_id)

        # 加载引擎
        do_parse_func = self._load_pipeline()

        try:
            # 读取文件
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # 格式转换
            if file_ext in [".png", ".jpg", ".jpeg"]:
                logger.info("🖼️  Converting image to PDF...")
                try:
                    pdf_bytes = img2pdf.convert(file_bytes)
                except Exception as e:
                    raise ValueError(f"Image conversion failed: {e}")
            else:
                pdf_bytes = file_bytes

            # 语言设置
            lang = options.get("lang", "auto")
            if lang == "auto": lang = "ch"

            # =================================================================
            # 【关键修复】使用安全文件名
            # =================================================================
            # 无论原文件名是什么（中文/特殊字符），内部处理时统一命名为 'result.pdf'
            # 这能避免 MinerU 内部处理路径时的编码问题
            safe_file_name = "result.pdf"
            logger.info(f"🚀 Processing as internal name: {safe_file_name}")

            # 调用 MinerU
            do_parse_func(
                output_dir=str(output_dir),
                pdf_file_names=[safe_file_name],  # 使用安全文件名
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[lang],
                
                backend=backend,
                parse_method=parse_method,
                server_url=server_url,
                
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                formula_enable=formula_enable,
                table_enable=table_enable,
                
                f_draw_layout_bbox=f_draw_layout_bbox,
                f_draw_span_bbox=f_draw_span_bbox,
                f_dump_md=f_dump_md,
                f_dump_middle_json=f_dump_middle_json,
                f_dump_model_output=f_dump_model_output,
                f_dump_orig_pdf=f_dump_orig_pdf,
                f_dump_content_list=f_dump_content_list
            )

            # =================================================================
            # 结果查找逻辑 (增强版)
            # =================================================================
            # MinerU 输出结构: {output_dir}/{safe_file_name}/auto/{safe_file_stem}.md
            # 例如: /app/data/output/.../result.pdf/auto/result.md
            
            # 1. 查找 Markdown
            md_files = list(output_dir.rglob("*.md"))
            
            # 2. 查找 Content JSON (降级)
            json_files = list(output_dir.rglob("*_content_list.json"))
            
            # 3. 查找 Layout PDF (底线)
            layout_files = list(output_dir.rglob("*_layout.pdf"))

            content = ""
            actual_output_dir = None
            json_path = None
            json_content = None
            md_path = None

            # 优先级 1: Markdown 存在
            if md_files:
                md_file = md_files[0]
                actual_output_dir = md_file.parent
                content = md_file.read_text(encoding="utf-8")
                md_path = str(md_file)
                logger.info(f"✅ Found MinerU MD output: {md_file.name}")

            # 优先级 2: JSON 存在 (MD 缺失)
            elif json_files:
                json_file = json_files[0]
                actual_output_dir = json_file.parent
                json_path = str(json_file)
                logger.warning(f"⚠️  MD missing, falling back to JSON: {json_file.name}")
                
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        json_content = data
                    # 简单尝试从 JSON 恢复文本
                    if isinstance(data, list):
                        content = "\n\n".join([b.get("text", "") for b in data if "text" in b])
                        logger.info("ℹ️  Recovered text from JSON content list")
                except Exception as e:
                    logger.error(f"❌ Failed to parse JSON: {e}")

            # 优先级 3: Layout PDF 存在 (文本识别完全失败)
            elif layout_files:
                layout_file = layout_files[0]
                # 这里的 parent 通常是 auto/ 目录
                actual_output_dir = layout_file.parent
                logger.warning(f"⚠️  Text extraction failed (no MD/JSON), but layout analysis succeeded: {layout_file.name}")
                content = "> ⚠️ Text extraction failed. Please check the layout visualization PDF in the output directory."
            
            else:
                # 彻底失败
                logger.error("❌ MinerU output directory structure:")
                for item in output_dir.rglob("*"):
                    logger.error(f"   {item}")
                raise FileNotFoundError(f"MinerU failed to generate any recognizable output in: {output_dir}")

            # 尝试补充 JSON 信息 (如果之前没加载)
            if actual_output_dir and not json_content:
                jsons = list(actual_output_dir.glob("*_content_list.json"))
                if jsons:
                    json_path = str(jsons[0])
                    try:
                        with open(jsons[0], "r", encoding="utf-8") as f:
                            json_content = json.load(f)
                    except: pass

            return {
                "markdown": content,
                "result_path": str(actual_output_dir) if actual_output_dir else str(output_dir),
                "json_path": json_path,
                "json_content": json_content,
                "markdown_file": md_path
            }

        finally:
            self.cleanup()


# 全局单例
_engine = None

def get_engine(vlm_api_base: str = None) -> MinerUPipelineEngine:
    global _engine
    if _engine is None:
        _engine = MinerUPipelineEngine(vlm_api_base=vlm_api_base)
    return _engine
