"""
PaddleOCR-VL-VLLM 解析引擎 (Optimized + Bidirectional Layout Support)
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API

功能增强:
1. [修复] 修复 res['res'] 类型不一致导致的 AttributeError 崩溃
2. [双向定位] 输出包含 bbox 的结构化数据 (json_content)
3. [资源管理] 智能显存休眠 (Auto-Sleep) 和自动唤醒 (Auto-Wakeup)
4. [稳定性] 强制单线程推理以解决 vLLM Tokenizer 竞态崩溃
5. [防崩溃] 增加 VLM NoneType 异常捕获与降级重试机制 (Fallback)
"""

import os
import gc
import json
import time
import requests
import traceback
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from threading import Lock
from loguru import logger

# ==============================================================================
# 🚨 全局环境配置 (必须在导入 paddle/paddlex 之前设置)
# ==============================================================================
# 1. 限制 PaddleX 内部推理并发数为 1，防止高并发请求冲垮 vLLM 的 Tokenizer
os.environ["PADDLEX_INFERENCE_PARALLEL_WORKER_NUM"] = "1"
# 2. 禁用模型源检查，加快启动速度 (内网环境必备)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# ==============================================================================

class PaddleOCRVLVLLMEngine:
    """
    PaddleOCR-VL-VLLM 解析引擎（支持双向定位数据输出）
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
            base_url = self.vllm_api_base.replace("/v1", "")
            health_url = f"{base_url}/health"
            try:
                requests.get(health_url, timeout=2)
                return True
            except:
                models_url = f"{self.vllm_api_base}/models"
                resp = requests.get(models_url, timeout=2)
                return resp.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️ VLLM service check failed: {e}")
            return False

    def _auto_sleep_monitor(self):
        """[后台线程] 监控空闲状态"""
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

            if not self._check_vllm_health():
                logger.error(f"❌ VLLM service unreachable at {self.vllm_api_base}")

            logger.info("📥 Loading PaddleOCR-VL-VLLM Pipeline (Auto-Wakeup)...")
            try:
                import paddle
                from paddleocr import PaddleOCRVL
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")

                self._pipeline = PaddleOCRVL(
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=self.vllm_api_base,
                )
                logger.info("✅ Pipeline loaded successfully")
                return self._pipeline
            except Exception as e:
                logger.error(f"❌ Pipeline load failed: {e}")
                raise

    def cleanup(self):
        """释放显存"""
        with self._lock:
            self._pipeline = None 
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
                gc.collect()
                logger.info("✅ GPU Memory released.")
            except:
                pass

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析文档入口并提取布局数据 (Phase 5 支持)
        """
        self.is_processing = True
        self.last_active_time = time.time()
        
        if self.is_offloaded:
            logger.info("🚀 New task received. Waking up engine...")
            self.is_offloaded = False

        try:
            file_path = Path(file_path)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"🤖 Processing: {file_path.name}")

            pipeline = self._load_pipeline()

            # =========================================================
            # 参数白名单过滤 (修复 NoneType error)
            # =========================================================
            allowed_params = {
                "use_doc_orientation_classify",
                "use_doc_unwarping",
                "use_layout_parsing",
                "use_chart_recognition",
                "use_seal_recognition",
                "use_ocr_for_image_block",
            }
            
            param_mapping = {
                "useDocOrientationClassify": "use_doc_orientation_classify",
                "useDocUnwarping": "use_doc_unwarping",
                "useLayoutDetection": "use_layout_parsing",
                "useChartRecognition": "use_chart_recognition",
                "useSealRecognition": "use_seal_recognition",
                "useOcrForImageBlock": "use_ocr_for_image_block",
            }

            predict_params = {"input": str(file_path)}
            
            for k, v in kwargs.items():
                target_key = param_mapping.get(k, k)
                if target_key in allowed_params:
                    predict_params[target_key] = v
                else:
                    logger.debug(f"ℹ️ Filtered param for VLLM mode: {k}={v}")
            
            # 强制默认值
            predict_params["use_layout_parsing"] = True
            predict_params["use_doc_orientation_classify"] = False
            predict_params["use_doc_unwarping"] = False

            # =========================================================
            # 🚨 [关键修复] 执行推理，增加防崩溃重试机制 (Fallback)
            # =========================================================
            try:
                # 强制转为 list，立即触发底层可能存在的 NoneType 错误
                output_generator = list(pipeline.predict(**predict_params))
            except Exception as e:
                logger.warning(f"⚠️ Standard prediction failed (likely VLM empty output): {e}")
                logger.info("🔄 Retrying with fallback parameters (disabling complex layout parsing)...")
                
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
            full_content_list = [] # [新增] 用于双向定位
            page_count = 0

            for res in output_generator:
                page_count += 1
                if res is None: continue

                page_dir = output_path / f"page_{page_count}"
                page_dir.mkdir(parents=True, exist_ok=True)

                # 1. 保存图片和原始 JSON
                try:
                    if hasattr(res, "save_to_img"): res.save_to_img(str(page_dir))
                    if hasattr(res, "save_to_json"): res.save_to_json(str(page_dir))
                except Exception as e:
                    logger.warning(f"Page {page_count} save error: {e}")

                # 2. [核心修复] 提取结构化数据 (BBox) 用于双向定位
                if hasattr(res, "json") and res.json:
                    json_list.append(res.json)
                    if isinstance(res.json, dict) and 'res' in res.json:
                        blocks = res.json['res']
                        
                        # [FIX] 严格类型检查，防止崩溃
                        if not isinstance(blocks, list):
                            # 如果是单个对象且有bbox，包装成列表
                            if isinstance(blocks, dict) and ('bbox' in blocks or 'layout_bbox' in blocks):
                                blocks = [blocks]
                            else:
                                # 可能是元数据（如 'input_path'），跳过
                                blocks = []

                        for block in blocks:
                            if not isinstance(block, dict): continue

                            clean_block = {
                                "id": len(full_content_list) + 1,
                                "page_idx": page_count - 1,
                                "type": block.get('type', 'text'),
                                "text": block.get('text', ''),
                                "bbox": block.get('layout_bbox') or block.get('bbox') or [],
                                "score": block.get('score', 0)
                            }
                            if clean_block['bbox']:
                                full_content_list.append(clean_block)

                # 3. 提取 Markdown
                page_md = ""
                if hasattr(res, "markdown") and res.markdown:
                    markdown_list_obj.append(res.markdown)
                    if isinstance(res.markdown, dict):
                        page_md = res.markdown.get('markdown_texts', '')
                    elif hasattr(res.markdown, 'markdown_texts'):
                        page_md = res.markdown.markdown_texts
                    else:
                        page_md = str(res.markdown)
                
                if page_md:
                    markdown_pages.append(page_md)
                
                logger.info(f"✅ Processed Page {page_count}")

            # 合并 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list_obj:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list_obj)
                except:
                    markdown_text = "\n\n---\n\n".join(markdown_pages)
            else:
                markdown_text = "\n\n---\n\n".join(markdown_pages)

            # 保存最终文件
            (output_path / "result.md").write_text(markdown_text, encoding="utf-8")
            
            # [关键] 构造 result.json
            final_json_data = full_content_list if full_content_list else {
                "total_pages": page_count,
                "pages": json_list
            }
            
            json_file = output_path / "result.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(output_path / "result.md"),
                "json_file": str(json_file),
                "json_content": full_content_list
            }

        except Exception as e:
            logger.error(f"❌ OCR Pipeline Error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.is_processing = False
            self.last_active_time = time.time()
            logger.info("🏁 Task finished. Model stays loaded (5min auto-sleep).")

# 全局单例
_engine = None

def get_engine(vllm_api_base: str = None, model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
