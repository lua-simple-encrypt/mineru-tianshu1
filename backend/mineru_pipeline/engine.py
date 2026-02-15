"""
MinerU Pipeline Engine
单例模式，每个进程只加载一次模型
使用 MinerU 处理 PDF 和图片

修复说明：
- [核心修复] 增加深度文本清洗 (_clean_markdown)，解决 HTML 转义、LaTeX 过度包装和模型幻觉标签
- [核心修复] 使用临时纯英文目录处理，规避中文路径问题
- [增强] 增加 VLLM 服务健康检查与自动等待机制
- [增强] 修复 Markdown 内容为空时的自动恢复逻辑
"""

import json
import os
import shutil
import tempfile
import time
import urllib.request
import urllib.error
import re        # <--- [新增] 正则表达式库
import html      # <--- [新增] HTML转义库
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

    def _wait_for_server(self, server_url: str, timeout: int = 60) -> bool:
        """等待 VLLM 服务就绪"""
        base_url = server_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
            
        health_url = f"{base_url}/v1/models"
        
        logger.info(f"⏳ Waiting for VLLM server at {base_url} (Timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        logger.info(f"✅ VLLM server is ready: {base_url}")
                        return True
            except (urllib.error.URLError, ConnectionRefusedError):
                pass
            except Exception as e:
                logger.debug(f"Health check warning: {e}")
            
            time.sleep(1)
            
        logger.warning(f"⚠️  VLLM server wait timed out after {timeout}s. Process may fail.")
        return False

    def _clean_markdown(self, text: str) -> str:
        """
        [关键功能] 深度清洗 Markdown 文本
        解决 HTML 转义、LaTeX 过度包装和模型幻觉问题
        """
        if not text:
            return ""

        # 1. HTML 反转义 (解决 &gt; -> >)
        # 必须先执行，因为有时候 &gt; 可能被包裹在 \mathrm{} 中
        text = html.unescape(text)

        # 2. 去除 LaTeX 的 \mathrm{} 包装 (解决 \mathrm{LVEDd} -> LVEDd)
        # 修改说明：
        # - 使用 (.*?) 非贪婪匹配，避免 [^\}]+ 无法匹配空内容或跨度过大的问题
        # - 添加 re.DOTALL 标志，确保 . 可以匹配换行符，处理跨行的 \mathrm{...}
        text = re.sub(r'\\mathrm\{(.*?)\}', r'\1', text, flags=re.DOTALL)

        # 3. 去除模型幻觉产生的 <del> 标签 (解决 <del>cm)
        text = text.replace('<del>', '').replace('</del>', '')

        return text

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
        处理文件 (增强版：临时目录 + 服务等待 + 深度清洗)
        """
        options = options or {}
        
        final_output_dir = Path(output_path)
        final_output_dir.mkdir(parents=True, exist_ok=True)

        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()

        # 1. 确定 Backend
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

        # 服务健康检查
        if "http-client" in backend and server_url:
            self._wait_for_server(server_url)

        # 2. 准备参数
        parse_method = options.get("method", "auto")
        if options.get("force_ocr"):
            parse_method = "ocr"

        formula_enable = options.get("formula_enable", True)
        table_enable = options.get("table_enable", True)
        
        f_draw_layout_bbox = options.get("draw_layout_bbox", True)      
        f_draw_span_bbox = options.get("draw_span_bbox", True)          
        f_dump_md = options.get("dump_markdown", True)                  
        f_dump_middle_json = options.get("dump_middle_json", True)      
        f_dump_model_output = options.get("dump_model_output", True)    
        f_dump_content_list = options.get("dump_content_list", True)    
        f_dump_orig_pdf = options.get("dump_orig_pdf", True)            

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

        do_parse_func = self._load_pipeline()

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

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

            # 使用临时纯英文目录处理
            with tempfile.TemporaryDirectory(prefix="mineru_proc_") as temp_dir:
                temp_work_dir = Path(temp_dir)
                logger.info(f"🛠️  Working in temp directory: {temp_work_dir}")
                
                safe_file_name = "result.pdf"
                
                do_parse_func(
                    output_dir=str(temp_work_dir),
                    pdf_file_names=[safe_file_name],
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

                # 结果提取与搬运
                generated_result_dir = temp_work_dir / "result"
                
                if not generated_result_dir.exists():
                    temp_md_files = list(temp_work_dir.rglob("*.md"))
                    if temp_md_files:
                        generated_result_dir = temp_md_files[0].parent.parent
                    else:
                        temp_json_files = list(temp_work_dir.rglob("*_content_list.json"))
                        if temp_json_files:
                             generated_result_dir = temp_json_files[0].parent.parent
                        else:
                             raise FileNotFoundError("Processing failed internally - No output generated")

                # 1. 读取内容并进行深度清洗
                content = ""
                json_content = None
                
                temp_md_files = list(generated_result_dir.rglob("*.md"))
                if temp_md_files:
                    md_file = temp_md_files[0]
                    raw_content = md_file.read_text(encoding="utf-8")
                    
                    # =========================================================
                    # 【核心修复】调用 _clean_markdown 进行深度清洗
                    # 解决 &gt;, \mathrm{}, <del> 等问题
                    # =========================================================
                    content = self._clean_markdown(raw_content)
                    
                    # 覆盖写入清洗后的内容
                    md_file.write_text(content, encoding="utf-8")
                    
                    logger.info(f"✅ Read and cleaned MD content: {len(content)} chars")
                
                temp_json_files = list(generated_result_dir.rglob("*_content_list.json"))
                if temp_json_files:
                    try:
                        with open(temp_json_files[0], "r", encoding="utf-8") as f:
                            json_content = json.load(f)
                    except: pass

                # 2. 如果 MD 为空，尝试从 JSON 恢复
                if not content.strip() and json_content:
                    logger.warning("⚠️  Markdown file is empty, attempting to recover text from JSON...")
                    recovered_text = []
                    if isinstance(json_content, list):
                        for block in json_content:
                            text = block.get("text", "")
                            # 恢复时也做清洗
                            text = self._clean_markdown(text)
                            recovered_text.append(text)
                    content = "\n\n".join(recovered_text)
                    
                    # 如果有 MD 文件，更新它
                    if temp_md_files:
                        temp_md_files[0].write_text(content, encoding="utf-8")
                        
                    logger.info(f"ℹ️  Recovered {len(content)} chars from JSON")

                # 3. 搬运文件
                logger.info(f"📦 Moving results from {generated_result_dir} to {final_output_dir}")
                if generated_result_dir.exists():
                    for src_path in generated_result_dir.rglob("*"):
                        if src_path.is_file():
                            rel_path = src_path.relative_to(generated_result_dir)
                            dest_path = final_output_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_path, dest_path)

                # 4. 返回路径
                final_md_path = None
                final_json_path = None
                
                final_mds = list(final_output_dir.rglob("*.md"))
                if final_mds:
                    final_md_path = str(final_mds[0])
                    # 覆盖写入清洗后的内容 (双重保险)
                    Path(final_md_path).write_text(content, encoding="utf-8")
                
                final_jsons = list(final_output_dir.rglob("*_content_list.json"))
                if final_jsons: final_json_path = str(final_jsons[0])

                if not content.strip():
                    layout_pdfs = list(final_output_dir.rglob("*_layout.pdf"))
                    if layout_pdfs:
                        content = "> ⚠️ Text extraction returned empty content. Please check layout PDF."
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
