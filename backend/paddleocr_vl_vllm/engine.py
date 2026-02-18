"""
PaddleOCR-VL-VLLM 解析引擎 (Ultimate Optimized Edition)
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API

功能增强:
1. [稳定性] 强制单线程推理以解决 vLLM Tokenizer "Already borrowed" 竞态崩溃问题
2. [防崩溃] 增加 VLM NoneType 异常捕获与降级重试机制 (Fallback)
3. [双向定位] 输出包含 bbox 的结构化数据 (json_content)，供前端双屏联动，已适配 block_order 排序
4. [资源管理] 智能显存休眠 (Auto-Sleep) 和自动唤醒 (Auto-Wakeup)
5. [高可用] 融合 MD 文件本地提取兜底与 PADDLEX_HOME 环境锁定
6. [并发控制] 拦截底层 HTTP 客户端强制串行化，彻底解决 vLLM 端 Tokenizer 崩溃
"""

import os
import gc
import json
import time
import requests
import traceback
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger

# ==============================================================================
# 🚨 全局环境配置 (必须在导入 paddle/paddlex 之前设置)
# ==============================================================================
# 1. 限制 PaddleX 内部推理并发数为 1，防止高并发请求冲垮 vLLM 的 Tokenizer
os.environ["PADDLEX_INFERENCE_PARALLEL_WORKER_NUM"] = "1"
os.environ["PADDLEX_API_MAX_WORKERS"] = "1"
# 2. 禁用模型源检查，加快启动速度 (内网环境必备)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ==============================================================================
# 🚨 终极防御：拦截 HTTP 客户端，限制 VLLM 高并发请求
# 彻底解决 vLLM Tokenizer "RuntimeError: Already borrowed" 崩溃问题
# ==============================================================================
try:
    import httpx

    # 全局信号量，强制完全串行，杜绝远端 Tokenizer 的 Rust 借用冲突
    _vllm_semaphore = threading.Semaphore(1)  

    # 1. Patch HTTPX (Sync - OpenAI SDK 底层使用)
    _original_httpx_send = httpx.Client.send
    def _throttled_httpx_send(self, request, *args, **kwargs):
        if "chat/completions" in str(request.url):
            with _vllm_semaphore:
                return _original_httpx_send(self, request, *args, **kwargs)
        return _original_httpx_send(self, request, *args, **kwargs)
    httpx.Client.send = _throttled_httpx_send

    # 2. Patch HTTPX (Async)
    _original_async_send = httpx.AsyncClient.send
    _async_semaphores = {}
    _async_sem_lock = threading.Lock()

    def _get_async_sem():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(1)
            
        with _async_sem_lock:
            if loop not in _async_semaphores:
                _async_semaphores[loop] = asyncio.Semaphore(1)
            return _async_semaphores[loop]

    async def _throttled_async_send(self, request, *args, **kwargs):
        if "chat/completions" in str(request.url):
            sem = _get_async_sem()
            async with sem:
                return await _original_async_send(self, request, *args, **kwargs)
        return await _original_async_send(self, request, *args, **kwargs)
    httpx.AsyncClient.send = _throttled_async_send

    # 3. Patch Requests (Sync - 兼容旧版或第三方库)
    _original_requests_send = requests.Session.send
    def _throttled_requests_send(self, request, **kwargs):
        if hasattr(request, 'url') and "chat/completions" in str(request.url):
            with _vllm_semaphore:
                return _original_requests_send(self, request, **kwargs)
        return _original_requests_send(self, request, **kwargs)
    requests.Session.send = _throttled_requests_send

    logger.info("🛡️ VLLM Network Throttling Patch applied successfully.")
except Exception as e:
    logger.warning(f"⚠️ Failed to patch HTTP clients: {e}")
# ==============================================================================

class PaddleOCRVLVLLMEngine:
    """
    PaddleOCR-VL-VLLM 解析引擎（企业级高可用版）
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

    def __init__(self, 
                 device: str = "cuda:0", 
                 vllm_api_base: str = None, 
                 model_name: str = "PaddleOCR-VL-1.5-0.9B"):
        """
        初始化引擎
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            # 优先使用传入参数，其次环境变量，最后默认 Docker 内部地址
            self.vllm_api_base = vllm_api_base or os.getenv("VLLM_API_BASE", "http://vllm-paddleocr:30023/v1")
            self.model_name = model_name

            # 提取 GPU ID
            if "cuda:" in device:
                try:
                    self.gpu_id = int(device.split(":")[-1])
                except ValueError:
                    self.gpu_id = 0
            else:
                self.gpu_id = 0

            self._check_gpu_availability()
            
            # =========================================================
            # [资源管理] 智能显存管理状态变量
            # =========================================================
            self.last_active_time = time.time()
            self.is_processing = False
            self.is_offloaded = True 
            self.idle_timeout = 300  # 5分钟无操作自动卸载

            # 启动监控线程
            self._monitor_thread = threading.Thread(target=self._auto_sleep_monitor, daemon=True)
            self._monitor_thread.start()

            self._initialized = True

            logger.info("🔧 PaddleOCR-VL-VLLM Engine Initialized")
            logger.info(f"   Device: {self.device} (Physical GPU: {self.gpu_id})")
            logger.info(f"   VLLM API: {self.vllm_api_base}")
            logger.info(f"   Concurrency: Serial Mode (Safe Network Patch Active)")
            logger.info(f"   Auto-Sleep: Enabled ({self.idle_timeout}s)")

    def _check_gpu_availability(self):
        try:
            import paddle
            if not paddle.is_compiled_with_cuda():
                logger.error("❌ PaddlePaddle is running on CPU! This model requires GPU.")
                return
            
            gpu_name = paddle.device.cuda.get_device_name(self.gpu_id)
            logger.info(f"✅ GPU Detected: {gpu_name}")
        except Exception:
            logger.warning("⚠️ Could not verify GPU status via PaddlePaddle")

    def _check_vllm_health(self) -> bool:
        """检查 VLLM 服务是否健康"""
        try:
            # 构造健康检查 URL (去除 /v1 后缀)
            base_url = self.vllm_api_base.replace("/v1", "")
            health_url = f"{base_url}/health"
            
            # 尝试请求 /health 或 /v1/models
            try:
                requests.get(health_url, timeout=2)
                return True
            except:
                # 回退尝试 models 接口
                models_url = f"{self.vllm_api_base}/models"
                resp = requests.get(models_url, timeout=2)
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️ VLLM service check failed: {e}")
            return False

    def _auto_sleep_monitor(self):
        """
        [后台线程] 监控空闲状态
        """
        while True:
            time.sleep(10)
            try:
                if self.is_processing or self.is_offloaded:
                    continue
                
                if time.time() - self.last_active_time > self.idle_timeout:
                    logger.info(f"💤 PaddleOCR-VLLM idle for {self.idle_timeout}s. Unloading pipeline...")
                    self.cleanup()
                    self.is_offloaded = True
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _load_pipeline(self):
        """延迟加载管道"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            # 1. 预检查 VLLM 服务
            if not self._check_vllm_health():
                logger.error(f"❌ VLLM service unreachable at {self.vllm_api_base}")
                logger.error("   Please ensure the 'vllm-paddleocr' container is running.")

            logger.info("=" * 60)
            logger.info("📥 Loading PaddleOCR-VL-VLLM Pipeline (Auto-Wakeup)...")
            logger.info("=" * 60)

            try:
                import paddle
                from paddleocr import PaddleOCRVL

                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")

                # 设置 PaddleX 主目录
                pdx_home = os.environ.get("PADDLEX_HOME", "/root/.paddlex")
                
                # 初始化管道
                self._pipeline = PaddleOCRVL(
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=self.vllm_api_base,
                )
                
                logger.info("✅ Pipeline loaded successfully (Serial Mode Active)")
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Pipeline load failed: {e}")
                logger.error(traceback.format_exc())
                raise

    def cleanup(self):
        """激进的显存清理"""
        with self._lock:
            self._pipeline = None # 释放引用
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
                gc.collect()
                logger.info("✅ VRAM released.")
            except:
                pass

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析文档入口 (增强版：自动唤醒 + 状态维护 + 防崩溃降级 + 双向定位提取)
        """
        # =========================================================
        # 1. 状态更新与自动唤醒
        # =========================================================
        self.is_processing = True
        self.last_active_time = time.time()
        
        if self.is_offloaded:
            logger.info("🚀 New task received. Waking up PaddleOCR-VLLM engine...")
            self.is_offloaded = False
            # _load_pipeline() 会自动重建

        try:
            file_path = Path(file_path)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"🤖 Processing: {file_path.name}")
            
            pipeline = self._load_pipeline()

            # 参数映射
            param_mapping = {
                "useDocOrientationClassify": "use_doc_orientation_classify",
                "useDocUnwarping": "use_doc_unwarping",
                "useLayoutDetection": "use_layout_parsing",
                "useChartRecognition": "use_chart_recognition",
                "useSealRecognition": "use_seal_recognition",
                "useOcrForImageBlock": "use_ocr_for_image_block",
                "layoutNms": "layout_nms",
                "markdownIgnoreLabels": "markdown_ignore_labels",
                "mergeTables": "merge_tables",
                "relevelTitles": "relevel_titles",
                "restructurePages": "restructure_pages",
                "minPixels": "min_pixels",
                "maxPixels": "max_pixels",
            }

            predict_params = {"input": str(file_path)}
            
            # 默认参数
            defaults = {
                "use_layout_parsing": True,
                "use_doc_orientation_classify": False,  # 默认关闭以防崩溃
                "use_doc_unwarping": False,
                "use_seal_recognition": True
            }
            
            # 填充参数
            for k, v in kwargs.items():
                if k in param_mapping:
                    predict_params[param_mapping[k]] = v
            
            for k, v in defaults.items():
                if k not in predict_params:
                    predict_params[k] = v

            # =========================================================
            # 🚨 2. 执行推理，增加防崩溃重试机制 (Fallback)
            # =========================================================
            try:
                # 强制转为 list，立即触发底层可能存在的 NoneType 错误
                output_generator = list(pipeline.predict(**predict_params))
            except Exception as e:
                logger.warning(f"⚠️ Standard prediction failed (likely VLM empty output/400 Error): {e}")
                logger.info("🔄 Retrying with fallback parameters (disabling complex layout parsing) in 2 seconds...")
                
                # 让远端 vLLM 服务器喘息恢复
                time.sleep(2)
                
                # 降级策略：关闭容易引起模型幻觉或空输出的高级版面分析
                predict_params["use_layout_parsing"] = False
                predict_params["use_chart_recognition"] = False
                predict_params["use_seal_recognition"] = False
                
                try:
                    output_generator = list(pipeline.predict(**predict_params))
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback prediction also failed: {fallback_e}")
                    raise RuntimeError(f"VLM Worker crashed on this document. Internal Error: {fallback_e}")

            markdown_pages = []
            markdown_list_obj = []
            json_list = []
            full_content_list = [] # 用于前端双向定位的高亮框数据
            page_count = 0

            for res in output_generator:
                page_count += 1
                
                # 🛡️ 防御性检查
                if res is None:
                    logger.error(f"❌ Page {page_count} returned None result")
                    continue

                page_output_dir = output_path / f"page_{page_count}"
                page_output_dir.mkdir(parents=True, exist_ok=True)

                # 保存中间图和JSON
                try:
                    if hasattr(res, "save_to_img"): res.save_to_img(str(page_output_dir))
                    if hasattr(res, "save_to_json"): res.save_to_json(str(page_output_dir))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save intermediate files for page {page_count}: {e}")

                # =========================================================
                # 3. [核心功能] 提取结构化数据 (BBox) 用于双向定位
                # =========================================================
                if hasattr(res, "json") and res.json:
                    json_list.append(res.json)
                    if isinstance(res.json, dict):
                        # 兼容 PaddleX 的不同返回格式 ('res' 或 'parsing_res_list')
                        blocks = res.json.get('res') or res.json.get('parsing_res_list') or []
                        
                        # 严格类型检查，防止崩溃
                        if not isinstance(blocks, list):
                            if isinstance(blocks, dict) and ('bbox' in blocks or 'layout_bbox' in blocks or 'block_bbox' in blocks):
                                blocks = [blocks]
                            else:
                                blocks = []

                        for block in blocks:
                            if not isinstance(block, dict): continue

                            clean_block = {
                                "id": len(full_content_list) + 1,
                                "page_idx": page_count - 1,
                                "type": block.get('type') or block.get('block_label') or 'text',
                                "text": block.get('text') or block.get('block_content') or '',
                                "bbox": block.get('layout_bbox') or block.get('block_bbox') or block.get('bbox') or [],
                                "score": block.get('score', 0),
                                "order": block.get('block_order') # 提取 block_order 保证前端排序正确
                            }
                            if clean_block['bbox']:
                                full_content_list.append(clean_block)

                # =========================================================
                # 4. 提取 Markdown (带本地文件读取兜底机制)
                # =========================================================
                if hasattr(res, "markdown") and res.markdown:
                    markdown_list_obj.append(res.markdown)

                page_md = ""
                try:
                    if hasattr(res, "markdown") and res.markdown:
                        if isinstance(res.markdown, dict):
                            page_md = res.markdown.get('markdown_texts', '') or res.markdown.get('text', '')
                        elif hasattr(res.markdown, 'markdown_texts'):
                            page_md = res.markdown.markdown_texts
                        else:
                            page_md = str(res.markdown)
                    elif hasattr(res, "str") and res.str:
                        page_md = str(res.str)
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting markdown from page {page_count}: {e}")

                if page_md:
                    markdown_pages.append(page_md)
                else:
                    # 兜底：尝试读取由于 save_to_markdown 生成的文件
                    try:
                         if hasattr(res, "save_to_markdown"):
                            res.save_to_markdown(str(page_output_dir))
                            saved = list(page_output_dir.glob("*.md"))
                            if saved:
                                markdown_pages.append(saved[0].read_text(encoding="utf-8"))
                    except:
                        pass
                
                logger.info(f"✅ Processed Page {page_count}")

            # 合并结果
            logger.info(f"🎉 Processing complete. Total pages: {page_count}")

            markdown_text = ""
            # 尝试使用官方合并算法
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list_obj:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list_obj)
                except Exception as e:
                    logger.warning(f"Official concat failed: {e}, falling back to simple join")
                    markdown_text = "\n\n---\n\n".join(markdown_pages)
            else:
                markdown_text = "\n\n---\n\n".join(markdown_pages)

            # 保存最终文件
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            
            # 构造 result.json (包含给前端定位用的 full_content_list)
            json_file = output_path / "result.json"
            final_json_data = full_content_list if full_content_list else {
                "pages": json_list, 
                "total_pages": page_count
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file),
                "json_content": final_json_data
            }

        except Exception as e:
            logger.error(f"❌ OCR Pipeline Critical Error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # =========================================================
            # [性能优化]
            # 移除强制 cleanup()，让模型保持加载状态
            # 更新时间戳，让后台线程在空闲5分钟后处理释放
            # =========================================================
            self.is_processing = False
            self.last_active_time = time.time()
            logger.info("🏁 Task finished. Pipeline remains loaded for fast reuse.")

# 全局单例
_engine = None

def get_engine(vllm_api_base: str = None, model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
