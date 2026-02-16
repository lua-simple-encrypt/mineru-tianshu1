"""
PaddleOCR-VL-VLLM 解析引擎 (Optimized + Bidirectional Layout Support)
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API

功能增强 (2026-02-15):
1. [双向定位] 输出包含 bbox 的结构化数据 (json_content)，支持前端点击跳转。
2. [资源管理] 包含智能显存休眠 (Auto-Sleep) 和自动唤醒 (Auto-Wakeup)。
3. [稳定性] 强制单线程推理以解决 vLLM Tokenizer 竞态崩溃。
"""

import os
import gc
import json
import time
import requests
import traceback
import threading
import uuid
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
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device
            self.vllm_api_base = vllm_api_base or os.getenv("VLLM_API_BASE", "http://vllm-paddleocr:30023/v1")
            self.model_name = model_name

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

            logger.info("=" * 60)
            logger.info("📥 Loading PaddleOCR-VL-VLLM Pipeline...")
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
                
                logger.info("✅ Pipeline loaded successfully")
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Pipeline load failed: {e}")
                logger.error(traceback.format_exc())
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
                logger.info("✅ VRAM released.")
            except:
                pass

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        执行解析并返回结构化数据（包含 bbox）
        """
        # 1. 自动唤醒
        self.is_processing = True
        self.last_active_time = time.time()
        
        if self.is_offloaded:
            logger.info("🚀 New task received. Waking up PaddleOCR-VLLM engine...")
            self.is_offloaded = False

        try:
            file_path = Path(file_path)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"🤖 Processing: {file_path.name}")
            
            pipeline = self._load_pipeline()

            # 参数映射 (保持与 Worker 一致)
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
            for k, v in kwargs.items():
                if k in param_mapping:
                    predict_params[param_mapping[k]] = v
            
            # 默认参数
            if "use_layout_parsing" not in predict_params: predict_params["use_layout_parsing"] = True
            if "use_doc_orientation_classify" not in predict_params: predict_params["use_doc_orientation_classify"] = False
            
            # 🚀 执行推理
            output_generator = pipeline.predict(**predict_params)

            markdown_pages = []
            markdown_list_obj = []
            
            # [关键] 用于存储所有页面的结构化数据 (含 bbox)
            full_content_list = [] 
            page_count = 0

            for res in output_generator:
                page_count += 1
                if res is None: continue

                page_output_dir = output_path / f"page_{page_count}"
                page_output_dir.mkdir(parents=True, exist_ok=True)

                # 1. 保存图片和 JSON
                if hasattr(res, "save_to_img"): res.save_to_img(str(page_output_dir))
                if hasattr(res, "save_to_json"): res.save_to_json(str(page_output_dir))

                # 2. 提取结构化数据 (包含 BBox)
                # PaddleX 的 res.json 通常包含 'res' 列表，里面有 layout_bbox 和 text
                if hasattr(res, "json") and res.json:
                    page_data = res.json
                    
                    # 尝试从 PaddleX 结果中提取 blocks
                    # 结构通常是: {'res': [{'bbox': [x,y,x,y], 'text': '...', 'type': '...'}, ...]}
                    if isinstance(page_data, dict) and 'res' in page_data:
                        blocks = page_data['res']
                        for block in blocks:
                            # 规范化 Block 数据供前端使用
                            clean_block = {
                                "id": len(full_content_list) + 1, # 全局唯一 ID
                                "page_idx": page_count - 1,       # 0-based page index
                                "type": block.get('type', 'text'),
                                "text": block.get('text', ''),
                                "bbox": block.get('layout_bbox') or block.get('bbox') or [], # 确保有坐标
                                "score": block.get('score', 0)
                            }
                            # 只有有坐标和内容的块才添加
                            if clean_block['bbox'] and (clean_block['text'] or clean_block['type'] in ['image', 'table']):
                                full_content_list.append(clean_block)

                # 3. 提取 Markdown
                if hasattr(res, "markdown") and res.markdown:
                    markdown_list_obj.append(res.markdown)
                    # 尝试提取字符串 markdown
                    if hasattr(res.markdown, 'markdown_texts'):
                        markdown_pages.append(res.markdown.markdown_texts)
                    elif isinstance(res.markdown, dict):
                        markdown_pages.append(res.markdown.get('markdown_texts', ''))
                    else:
                        markdown_pages.append(str(res.markdown))
                
                logger.info(f"✅ Processed Page {page_count}")

            # 合并 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages") and markdown_list_obj:
                try:
                    markdown_text = pipeline.concatenate_markdown_pages(markdown_list_obj)
                except:
                    markdown_text = "\n\n---\n\n".join(markdown_pages)
            else:
                markdown_text = "\n\n---\n\n".join(markdown_pages)

            # 保存结果
            (output_path / "result.md").write_text(markdown_text, encoding="utf-8")
            
            # [关键] 生成 content_list.json (扁平化结构，供前端定位使用)
            # 如果 full_content_list 为空（某些模型模式下），尝试用 json_list 兜底，或者前端做兼容
            final_json_data = full_content_list
            
            # 保存 detailed JSON
            json_file = output_path / "result.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(final_json_data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(output_path / "result.md"),
                "json_file": str(json_file),
                # 返回 json_content 给 worker，worker 会将其存入 DB
                "json_content": final_json_data 
            }

        except Exception as e:
            logger.error(f"❌ OCR Pipeline Critical Error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.is_processing = False
            self.last_active_time = time.time()
            logger.info("🏁 Task finished. Pipeline remains loaded.")

# 全局单例
_engine = None

def get_engine(vllm_api_base: str = None, model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
