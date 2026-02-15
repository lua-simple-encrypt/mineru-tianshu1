"""
PaddleOCR-VL-VLLM 解析引擎 (Optimized)
单例模式，每个进程只加载一次基础版面识别模型, OCR部分调用配置的API

🚨 CRITICAL FIX APPLIED: 
强制单线程推理以解决 vLLM Tokenizer "Already borrowed" 竞态崩溃问题。

参考文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html
"""

import os
import gc
import json
import time
import requests
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
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
    PaddleOCR-VL-VLLM 解析引擎（企业级优化版）

    特性：
    - 稳定优先：强制串行请求，消除底层 Rust Tokenizer 崩溃
    - 自动多语言：支持 109+ 种语言自动识别
    - 显存保护：流式处理 + 激进的 GC 策略
    - 故障隔离：预检查 VLLM 服务状态
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
            self._initialized = True

            logger.info("🔧 PaddleOCR-VL-VLLM Engine Initialized")
            logger.info(f"   Device: {self.device} (Physical GPU: {self.gpu_id})")
            logger.info(f"   VLLM API: {self.vllm_api_base}")
            logger.info(f"   Concurrency: Serial Mode (Safe)")

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
                # 这里不抛出异常，尝试继续加载，因为有时网络可能短暂波动

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
                
                logger.info("✅ Pipeline loaded successfully (Serial Mode Active)")
                return self._pipeline

            except Exception as e:
                logger.error(f"❌ Pipeline load failed: {e}")
                logger.error(traceback.format_exc())
                raise

    def cleanup(self):
        """激进的显存清理"""
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
            gc.collect()
        except:
            pass

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        解析文档入口
        """
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

        try:
            # 🚀 执行推理
            # 注意：由于我们在文件头设置了 PARALLEL_WORKER_NUM=1
            # 这里即使是大文件，也会一页页串行发送给 VLLM，不会再触发 400 错误
            output_generator = pipeline.predict(**predict_params)

            markdown_pages = []
            markdown_list_obj = []
            json_list = []
            page_count = 0

            for res in output_generator:
                page_count += 1
                
                # 🛡️ 防御性检查：防止 NoneType 错误
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

                # 收集数据
                if hasattr(res, "json") and res.json:
                    json_list.append(res.json)

                if hasattr(res, "markdown") and res.markdown:
                    markdown_list_obj.append(res.markdown)

                # 提取 Markdown 文本
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
            
            json_file = output_path / "result.json"
            combined_json = {"pages": json_list, "total_pages": page_count}
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(combined_json, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file),
            }

        except Exception as e:
            logger.error(f"❌ OCR Pipeline Critical Error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup()

# 全局单例
_engine = None

def get_engine(vllm_api_base: str = None, model_name: str = "PaddleOCR-VL-1.5-0.9B") -> PaddleOCRVLVLLMEngine:
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine(vllm_api_base=vllm_api_base, model_name=model_name)
    return _engine
