"""
MinerU Pipeline Engine
单例模式，每个进程只加载一次模型
使用 MinerU 处理 PDF 和图片

修复说明：
- [关键] 使用临时纯英文目录进行处理，彻底规避中文路径导致的写入失败/内容为空问题
- [关键] 修复 Markdown 内容为空时的自动恢复逻辑 (从 JSON 重建)
- [修复] 强制使用安全文件名 'result.pdf' 进行内部处理
- [修复] 正确透传 backend/server_url 参数以启用 VLLM 加速
"""

import json
import os
import shutil
import tempfile
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
        处理文件 (增强版：使用临时目录规避路径问题)
        """
        options = options or {}
        
        # 用户指定的最终输出目录 (可能包含中文)
        final_output_dir = Path(output_path)
        final_output_dir.mkdir(parents=True, exist_ok=True)

        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()

        # =========================================================================
        # 1. 确定 Backend (处理模式)
        # =========================================================================
        user_backend = options.get("parse_mode", "pipeline")
        if user_backend == "auto":
            user_backend = "pipeline"

        backend = user_backend
        server_url = options.get("server_url")

        # 智能切换 VLLM 加速
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
        
        # 输出控制 (默认开启以确保调试文件生成)
        f_draw_layout_bbox = options.get("draw_layout_bbox", True)      
        f_draw_span_bbox = options.get("draw_span_bbox", True)          
        f_dump_md = options.get("dump_markdown", True)                  
        f_dump_middle_json = options.get("dump_middle_json", True)      
        f_dump_model_output = options.get("dump_model_output", True)    
        f_dump_content_list = options.get("dump_content_list", True)    
        f_dump_orig_pdf = options.get("dump_orig_pdf", True)            

        # 页面范围
        start_page_id = options.get("start_page_id", 0)
        end_page_id = options.get("end_page_id", None)
        
        try: start_page_id = int(start_page_id)
        except: start_page_id = 0
        
        try: 
            if end_page_id is not None and str(end_page_id).strip() != "": 
                end_page_id = int(end_page_id)
                if end_page_id == -1: end_page_id = None
            else: end_page_id = None
        except: end_page_id = None

        # 加载引擎
        do_parse_func = self._load_pipeline()

        try:
            # 读取源文件
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

            lang = options.get("lang", "auto")
            if lang == "auto": lang = "ch"

            # =================================================================
            # 【核心修复】使用临时纯英文目录处理
            # =================================================================
            # 创建临时目录，确保路径不含任何中文或特殊字符
            # 这能彻底解决 MinerU 底层库因为路径编码问题导致的写入失败或空文件
            with tempfile.TemporaryDirectory(prefix="mineru_proc_") as temp_dir:
                temp_work_dir = Path(temp_dir)
                logger.info(f"🛠️  Working in temp directory: {temp_work_dir}")
                
                # 强制使用安全文件名
                safe_file_name = "result.pdf"
                
                # 调用 MinerU 处理
                do_parse_func(
                    output_dir=str(temp_work_dir), # 输出到临时目录
                    pdf_file_names=[safe_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=[lang],
                    
                    # 关键参数透传
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

                # =============================================================
                # 结果提取与搬运
                # =============================================================
                # MinerU 输出结构: {temp_work_dir}/result/auto/result.md
                # 注意：result 来自 safe_file_name 的 stem
                generated_result_dir = temp_work_dir / "result"
                
                if not generated_result_dir.exists():
                    logger.error("❌ MinerU failed to generate output directory in temp workspace")
                    raise FileNotFoundError("Processing failed internally")

                # 1. 在临时目录中读取内容 (此时路径绝对安全)
                content = ""
                json_content = None
                
                # 查找 Markdown
                temp_md_files = list(generated_result_dir.rglob("*.md"))
                if temp_md_files:
                    md_file = temp_md_files[0]
                    content = md_file.read_text(encoding="utf-8")
                    logger.info(f"✅ Read MD content: {len(content)} chars")
                
                # 查找 JSON
                temp_json_files = list(generated_result_dir.rglob("*_content_list.json"))
                if temp_json_files:
                    try:
                        with open(temp_json_files[0], "r", encoding="utf-8") as f:
                            json_content = json.load(f)
                    except: pass

                # 【修复内容为空】如果 MD 为空，尝试从 JSON 恢复
                if not content.strip() and json_content:
                    logger.warning("⚠️  Markdown file is empty, attempting to recover text from JSON...")
                    recovered_text = []
                    if isinstance(json_content, list):
                        for block in json_content:
                            if "text" in block:
                                recovered_text.append(block["text"])
                    content = "\n\n".join(recovered_text)
                    logger.info(f"ℹ️  Recovered {len(content)} chars from JSON")

                # 2. 将结果文件搬运到用户指定的 final_output_dir
                # 将 result/auto 下的所有文件复制过去
                # 或者直接把 result 文件夹里的内容复制到 final_output_dir
                logger.info(f"📦 Moving results to: {final_output_dir}")
                
                for src_path in generated_result_dir.rglob("*"):
                    if src_path.is_file():
                        # 计算相对路径，保持目录结构 (例如 auto/images/1.jpg)
                        rel_path = src_path.relative_to(generated_result_dir)
                        dest_path = final_output_dir / rel_path
                        
                        # 确保目标父目录存在
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 复制文件
                        shutil.copy2(src_path, dest_path)

                # =============================================================
                # 返回最终结果路径
                # =============================================================
                # 重新定位最终目录下的关键文件
                final_md_path = None
                final_json_path = None
                
                # 查找最终目录下的 MD
                final_mds = list(final_output_dir.rglob("*.md"))
                if final_mds:
                    final_md_path = str(final_mds[0])
                
                # 查找最终目录下的 JSON
                final_jsons = list(final_output_dir.rglob("*_content_list.json"))
                if final_jsons:
                    final_json_path = str(final_jsons[0])

                if not content.strip():
                    # 最后一道防线：检查是否有布局 PDF
                    layout_pdfs = list(final_output_dir.rglob("*_layout.pdf"))
                    if layout_pdfs:
                        content = "> ⚠️ Text extraction returned empty content. Please check layout PDF."
                        logger.warning("⚠️  Returning empty content warning.")
                    else:
                        raise FileNotFoundError("No valid content generated.")

                return {
                    "markdown": content,
                    "result_path": str(final_output_dir),
                    "markdown_file": final_md_path,
                    "json_path": final_json_path,
                    "json_content": json_content
                }

        except Exception as e:
            logger.error(f"❌ Pipeline processing failed: {e}")
            # 尝试打印错误堆栈
            import traceback
            logger.debug(traceback.format_exc())
            raise

        finally:
            self.cleanup()


# 全局单例
_engine = None

def get_engine(vlm_api_base: str = None) -> MinerUPipelineEngine:
    global _engine
    if _engine is None:
        _engine = MinerUPipelineEngine(vlm_api_base=vlm_api_base)
    return _engine
