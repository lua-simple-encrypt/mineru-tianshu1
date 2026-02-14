"""
MinerU Tianshu - API Server
天枢 API 服务器

企业级 AI 数据预处理平台
支持文档、图片、音频、视频等多模态数据处理
提供 RESTful API 接口用于任务提交、查询和管理
企业级认证授权: JWT Token + API Key + SSO
"""

import json
import os
import re
import uuid
import mimetypes  # ✅ [新增] 用于自动识别文件类型
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

# 导入认证模块
from auth import (
    User,
    Permission,
    get_current_active_user,
    require_permission,
)
from auth.auth_db import AuthDB
from auth.routes import router as auth_router
from task_db import TaskDB

# ✅ [优化] 预注册 MIME 类型，防止精简环境识别失败导致浏览器强制下载
mimetypes.add_type('application/pdf', '.pdf')
mimetypes.add_type('image/png', '.png')
mimetypes.add_type('image/jpeg', '.jpg')
mimetypes.add_type('image/jpeg', '.jpeg')
mimetypes.add_type('text/markdown', '.md')
mimetypes.add_type('application/json', '.json')

# 初始化 FastAPI 应用
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持文档、图片、音频、视频等多模态数据处理 | 企业级认证授权",
    version="2.0.0",
    # 不设置 servers，让 FastAPI 自动根据请求的 Host 生成
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录（backend 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent

# 初始化数据库
# 确保使用环境变量中的数据库路径（与 Worker 保持一致）
db_path_env = os.getenv("DATABASE_PATH")
if db_path_env:
    db_path = str(Path(db_path_env).resolve())
    logger.info(f"📊 API Server using DATABASE_PATH: {db_path_env} -> {db_path}")
    db = TaskDB(db_path)
else:
    logger.warning("⚠️  DATABASE_PATH not set in API Server, using default")
    # Docker 环境: /app/data/db/mineru_tianshu.db
    # 本地环境: ./data/db/mineru_tianshu.db
    default_db_path = PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db"
    default_db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = str(default_db_path.resolve())
    logger.info(f"📊 Using default database path: {db_path}")
    db = TaskDB(db_path)
auth_db = AuthDB()

# 注册认证路由
app.include_router(auth_router)

# ==============================================================================
# 目录配置 (Output & Upload)
# ==============================================================================

# 1. 配置输出目录（使用共享目录，Docker 环境可访问）
output_path_env = os.getenv("OUTPUT_PATH")
if output_path_env:
    OUTPUT_DIR = Path(output_path_env).resolve()
else:
    # Docker 环境: /app/output
    # 本地环境: ./data/output
    OUTPUT_DIR = (PROJECT_ROOT / "data" / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"📁 Output directory: {OUTPUT_DIR}")

# 2. 配置上传目录 (修改默认为 input)
upload_path_env = os.getenv("UPLOAD_PATH")
if upload_path_env:
    UPLOAD_DIR = Path(upload_path_env).resolve()
else:
    # Docker 环境: /app/input (如果不设置环境变量)
    # 本地环境: ./input (项目根目录下的 input 目录)
    UPLOAD_DIR = (PROJECT_ROOT / "input").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")


# 注意：此函数已废弃，Worker 已自动上传图片到 RustFS 并替换 URL
# 保留此函数仅用于向后兼容（处理旧任务或 RustFS 失败的情况）
def process_markdown_images_legacy(md_content: str, image_dir: Path, result_path: str):
    """
    【向后兼容】处理 Markdown 中的图片引用

    Worker 已自动上传图片到 RustFS 并替换 URL，此函数仅用于向后兼容。
    如果检测到图片路径不是 URL，则转换为本地静态文件服务 URL。
    """
    # 检查是否已经包含 RustFS URL
    if "http://" in md_content or "https://" in md_content:
        logger.debug("✅ Markdown already contains URLs (RustFS uploaded)")
        return md_content

    # 如果没有图片目录，直接返回
    if not image_dir.exists():
        logger.debug("ℹ️  No images directory, skipping processing")
        return md_content

    # 兼容模式：转换相对路径为本地 URL
    logger.warning("⚠️  Images not uploaded to RustFS, using local URLs (legacy mode)")

    def replace_image_path(match):
        """替换图片路径为本地 URL"""
        full_match = match.group(0)
        # 提取图片路径（Markdown 或 HTML）
        if "![" in full_match:
            # Markdown: ![alt](path)
            image_path = match.group(2)
            alt_text = match.group(1)
        else:
            # HTML: <img src="path">
            image_path = match.group(2)
            alt_text = "Image"

        # 如果已经是 URL，跳过
        if image_path.startswith("http"):
            return full_match

        # 生成本地静态文件 URL
        try:
            image_filename = Path(image_path).name
            output_dir_str = str(OUTPUT_DIR).replace("\\", "/")
            result_path_str = result_path.replace("\\", "/")

            if result_path_str.startswith(output_dir_str):
                relative_path = result_path_str[len(output_dir_str) :].lstrip("/")
                # ✅ [修复] url 编码需保留正斜杠，防止 404
                encoded_relative_path = quote(relative_path, safe="/")
                encoded_filename = quote(image_filename, safe="/")
                
                # 统一使用 /api/v1 前缀，稍后通过 Router 注册兼容 Nginx
                static_url = f"/api/v1/files/output/{encoded_relative_path}/images/{encoded_filename}"

                # 返回替换后的内容
                if "![" in full_match:
                    return f"![{alt_text}]({static_url})"
                else:
                    return full_match.replace(image_path, static_url)
        except Exception as e:
            logger.error(f"❌ Failed to generate local URL: {e}")

        return full_match

    try:
        # 匹配 Markdown 和 HTML 图片
        md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        html_pattern = r'<img\s+([^>]*\s+)?src="([^"]+)"([^>]*)>'

        new_content = re.sub(md_pattern, replace_image_path, md_content)
        new_content = re.sub(html_pattern, replace_image_path, new_content)
        return new_content
    except Exception as e:
        logger.error(f"❌ Failed to process images: {e}")
        return md_content


@app.get("/", tags=["系统信息"])
async def root():
    """API根路径"""
    return {
        "service": "MinerU Tianshu",
        "version": "2.0.0",
        "description": "天枢 - 企业级 AI 数据预处理平台",
        "features": "文档、图片、音频、视频等多模态数据处理",
        "docs": "/docs",
    }


# ============================================================================
# 创建 API Router (核心修复：解决 Nginx 路径剥离问题)
# ============================================================================
# 所有的业务接口都挂载到 router 上，然后注册两次：
# 1. /api/v1 (完整路径)
# 2. /v1 (Nginx 剥离后路径)
router = APIRouter()


@router.post("/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(..., description="文件: PDF/图片/Office/HTML/音频/视频等多种格式"),
    backend: str = Form(
        "auto",
        description="处理后端: pipeline, hybrid-auto-engine, vlm-auto-engine, hybrid-http-client, vlm-http-client, paddleocr-vl, etc.",
    ),
    lang: str = Form("auto", description="语言: ch/en/auto..."),
    method: str = Form("auto", description="解析方法: auto/txt/ocr"),
    formula_enable: bool = Form(True, description="是否启用公式识别"),
    table_enable: bool = Form(True, description="是否启用表格识别"),
    priority: int = Form(0, description="优先级，数字越大越优先"),
    
    # === 新增参数 ===
    start_page: Optional[int] = Form(None, description="起始页码（从0开始）"),
    end_page: Optional[int] = Form(None, description="结束页码"),
    # force_ocr 保留兼容，但建议使用 method='ocr'
    force_ocr: bool = Form(False, description="[兼容旧版] 是否强制使用OCR"),
    
    # 远程服务参数
    server_url: Optional[str] = Form(None, description="远程服务器地址 (仅 Client 模式需要)"),

    # MinerU 详细调试/输出选项 (对应前端 Advanced Settings)
    draw_layout_bbox: bool = Form(True, description="绘制布局边框 (_layout.pdf)"),
    draw_span_bbox: bool = Form(True, description="绘制文本边框 (_span.pdf)"),
    dump_markdown: bool = Form(True, description="输出 Markdown"),
    dump_middle_json: bool = Form(True, description="输出中间 JSON"),
    dump_model_output: bool = Form(True, description="输出模型原始数据"),
    dump_content_list: bool = Form(True, description="输出内容列表"),
    dump_orig_pdf: bool = Form(True, description="保存原始/截取 PDF"),
    
    # 旧版参数兼容 (Worker 会做映射)
    draw_layout: bool = Form(True, description="[兼容旧版] 是否绘制布局边框"),
    draw_span: bool = Form(True, description="[兼容旧版] 是否绘制文本Span边框"),
    
    # 视频处理专用参数
    keep_audio: bool = Form(False, description="视频处理时是否保留提取的音频文件"),
    enable_keyframe_ocr: bool = Form(False, description="是否启用视频关键帧OCR识别（实验性功能）"),
    ocr_backend: str = Form("paddleocr-vl", description="关键帧OCR引擎: paddleocr-vl"),
    keep_keyframes: bool = Form(False, description="是否保留提取的关键帧图像"),
    
    # 音频处理专用参数
    enable_speaker_diarization: bool = Form(
        False, description="是否启用说话人分离（音频多说话人识别，需要额外下载 Paraformer 模型）"
    ),
    
    # 水印去除专用参数
    remove_watermark: bool = Form(False, description="是否启用水印去除（支持 PDF/图片）"),
    watermark_conf_threshold: float = Form(0.35, description="水印检测置信度阈值（0.0-1.0，推荐 0.35）"),
    watermark_dilation: int = Form(10, description="水印掩码膨胀大小（像素，推荐 10）"),
    
    # Office 文件转 PDF 参数
    convert_office_to_pdf: bool = Form(
        False,
        description="是否将 Office 文件转换为 PDF 后再处理（图片提取更完整，但速度较慢）"
    ),
    
    # 认证依赖
    current_user: User = Depends(require_permission(Permission.TASK_SUBMIT)),
):
    """
    提交文档解析任务
    """
    try:
        # 生成唯一的文件名（避免冲突）
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = UPLOAD_DIR / unique_filename

        # 流式写入文件到磁盘，避免高内存使用
        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(1 << 23)  # 8MB chunks
                if not chunk:
                    break
                temp_file.write(chunk)

        # 构建处理选项
        options = {
            "lang": lang,
            "method": method,
            "formula_enable": formula_enable,
            "table_enable": table_enable,
            "start_page": start_page,
            "end_page": end_page,
            "force_ocr": force_ocr,
            "server_url": server_url,
            "draw_layout_bbox": draw_layout_bbox,
            "draw_span_bbox": draw_span_bbox,
            "dump_markdown": dump_markdown,
            "dump_middle_json": dump_middle_json,
            "dump_model_output": dump_model_output,
            "dump_content_list": dump_content_list,
            "dump_orig_pdf": dump_orig_pdf,
            "draw_layout": draw_layout,
            "draw_span": draw_span,
            "keep_audio": keep_audio,
            "enable_keyframe_ocr": enable_keyframe_ocr,
            "ocr_backend": ocr_backend,
            "keep_keyframes": keep_keyframes,
            "enable_speaker_diarization": enable_speaker_diarization,
            "remove_watermark": remove_watermark,
            "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation,
            "convert_office_to_pdf": convert_office_to_pdf,
        }

        # ✅ [修复 Bug 3]：自动检测环境变量开启 RustFS，默认传递给 Worker
        options["upload_images"] = os.getenv("RUSTFS_ENABLED", "true").lower() == "true"

        # 创建任务
        task_id = db.create_task(
            file_name=file.filename,
            file_path=str(temp_file_path),
            backend=backend,
            options=options,
            priority=priority,
            user_id=current_user.user_id,
        )

        logger.info(f"✅ Task submitted: {task_id} - {file.filename}")
        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": "Task submitted successfully",
            "file_name": file.filename,
            "user_id": current_user.user_id,
            "created_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str,
    upload_images: bool = Query(False, description="【已废弃】图片已自动上传到 RustFS"),
    format: str = Query("markdown", description="返回格式: markdown(默认)/json/both"),
    current_user: User = Depends(get_current_active_user),
):
    """
    查询任务状态和详情
    """
    task = db.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 权限检查
    if not current_user.has_permission(Permission.TASK_VIEW_ALL):
        if task.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Permission denied: You can only view your own tasks")

    # === 构建源文件访问 URL ===
    source_url = None
    if task.get("file_path"):
        try:
            source_filename = Path(task["file_path"]).name
            encoded_source_filename = quote(source_filename)
            # 始终返回完整路径 /api/v1/... 前端使用方便
            source_url = f"/api/v1/files/upload/{encoded_source_filename}"
        except Exception as e:
            logger.warning(f"Failed to generate source_url: {e}")

    response = {
        "success": True,
        "task_id": task_id,
        "status": task["status"],
        "file_name": task["file_name"],
        "source_url": source_url,
        "backend": task["backend"],
        "priority": task["priority"],
        "error_message": task["error_message"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "user_id": task.get("user_id"),
    }

    if not task.get("is_parent"):
        response["worker_id"] = task.get("worker_id")
        response["retry_count"] = task.get("retry_count")

    if task.get("is_parent"):
        child_count = task.get("child_count", 0)
        child_completed = task.get("child_completed", 0)
        response["is_parent"] = True
        response["subtask_progress"] = {
            "total": child_count,
            "completed": child_completed,
            "percentage": round(child_completed / child_count * 100, 1) if child_count > 0 else 0,
        }
        try:
            children = db.get_child_tasks(task_id)
            response["subtasks"] = [
                {
                    "task_id": child["task_id"],
                    "status": child["status"],
                    "chunk_info": json.loads(child.get("options", "{}")).get("chunk_info"),
                    "error_message": child.get("error_message"),
                }
                for child in children
            ]
        except Exception as e:
            logger.warning(f"⚠️  Failed to load subtasks: {e}")

    if task["status"] == "completed":
        if not task["result_path"]:
            response["data"] = None
            response["message"] = "Task completed but result files have been cleaned up"
            return response

        result_dir = Path(task["result_path"])
        if result_dir.exists():
            md_files = list(result_dir.rglob("*.md"))
            json_files = [
                f for f in result_dir.rglob("*.json")
                if not f.parent.name.startswith("page_")
                and (f.name in ["content.json", "result.json"] or "_content_list.json" in f.name)
            ]
            
            if md_files or json_files:
                try:
                    response["data"] = {}
                    response["data"]["json_available"] = len(json_files) > 0
                    
                    pdf_files = list(result_dir.rglob("*.pdf"))
                    preview_pdf = None
                    for pdf in pdf_files:
                        if "_layout.pdf" in pdf.name:
                            preview_pdf = pdf
                            break
                    if not preview_pdf:
                         for pdf in pdf_files:
                             if "_span.pdf" in pdf.name:
                                 preview_pdf = pdf
                                 break
                    if not preview_pdf:
                        for pdf in pdf_files:
                            if not pdf.name.startswith("page_"):
                                preview_pdf = pdf
                                break

                    if preview_pdf:
                        try:
                             rel_path = preview_pdf.relative_to(OUTPUT_DIR)
                             encoded_path = quote(str(rel_path).replace("\\", "/"), safe="/")
                             response["data"]["pdf_path"] = encoded_path
                        except ValueError:
                             pass

                    if format in ["markdown", "both"] and md_files:
                        md_file = None
                        for f in md_files:
                            if f.name == "result.md":
                                md_file = f
                                break
                        if not md_file:
                            md_file = md_files[0]

                        image_dir = md_file.parent / "images"
                        with open(md_file, "r", encoding="utf-8") as f:
                            md_content = f.read()

                        if image_dir.exists() and ("http://" not in md_content and "https://" not in md_content):
                            md_content = process_markdown_images_legacy(md_content, image_dir, task["result_path"])

                        response["data"]["markdown_file"] = md_file.name
                        response["data"]["content"] = md_content
                        response["data"]["has_images"] = image_dir.exists()

                    if format in ["json", "both"] and json_files:
                        import json as json_lib
                        json_file = json_files[0]
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                json_content = json_lib.load(f)
                            response["data"]["json_file"] = json_file.name
                            response["data"]["json_content"] = json_content
                        except Exception as json_e:
                            logger.warning(f"⚠️  Failed to load JSON: {json_e}")
                    elif format == "json" and not json_files:
                        response["data"]["message"] = "JSON format not available for this backend"

                    if not response["data"]:
                        response["data"] = None

                except Exception as e:
                    logger.error(f"❌ Failed to read content: {e}")
                    response["data"] = None
        else:
            logger.error(f"❌ Result directory does not exist: {result_dir}")

    return response


@router.delete("/tasks/{task_id}", tags=["任务管理"])
async def cancel_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    """
    取消任务（仅限 pending 状态）
    """
    task = db.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not current_user.has_permission(Permission.TASK_DELETE_ALL):
        if task.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Permission denied: You can only cancel your own tasks")

    if task["status"] == "pending":
        db.update_task_status(task_id, "cancelled")
        file_path = Path(task["file_path"])
        if file_path.exists():
            file_path.unlink()
        logger.info(f"⏹️  Task cancelled: {task_id} by user {current_user.username}")
        return {"success": True, "message": "Task cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in {task['status']} status")


@router.get("/queue/stats", tags=["队列管理"])
async def get_queue_stats(current_user: User = Depends(require_permission(Permission.QUEUE_VIEW))):
    """
    获取队列统计信息
    """
    stats = db.get_queue_stats()
    return {
        "success": True,
        "stats": stats,
        "total": sum(stats.values()),
        "timestamp": datetime.now().isoformat(),
        "user": current_user.username,
    }


@router.get("/queue/tasks", tags=["队列管理"])
async def list_tasks(
    status: Optional[str] = Query(None, description="筛选状态"),
    limit: int = Query(100, description="返回数量限制", le=1000),
    page: int = Query(1, ge=1, description="页码"),  
    page_size: int = Query(20, ge=1, le=100, description="每页数量"), 
    backend: Optional[str] = Query(None, description="筛选后端引擎"), 
    search: Optional[str] = Query(None, description="搜索文件名或任务ID"), 
    current_user: User = Depends(get_current_active_user),
):
    """
    获取任务列表
    """
    can_view_all = current_user.has_permission(Permission.TASK_VIEW_ALL)
    conditions = []
    params = []

    if not can_view_all:
        conditions.append("user_id = ?")
        params.append(current_user.user_id)

    if status:
        conditions.append("status = ?")
        params.append(status)
    if backend:
        conditions.append("backend = ?")
        params.append(backend)
    
    if search:
        search = search.strip()
        conditions.append("(file_name LIKE ? OR task_id = ?)")
        params.append(f"%{search}%")
        params.append(search)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    offset = (page - 1) * page_size

    with db.get_cursor() as cursor:
        count_sql = f"SELECT COUNT(*) FROM tasks{where_clause}"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()[0]

        query_params = params + [page_size, offset]
        data_sql = f"""
            SELECT * FROM tasks
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_sql, query_params)
        tasks = [dict(row) for row in cursor.fetchall()]

    return {
        "success": True, 
        "total": total,
        "page": page,
        "page_size": page_size,
        "count": len(tasks),
        "tasks": tasks, 
        "can_view_all": can_view_all
    }


@router.post("/admin/cleanup", tags=["系统管理"])
async def cleanup_old_tasks(
    days: int = Query(7, description="清理N天前的任务"),
    current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE)),
):
    """
    清理旧任务（管理接口）
    """
    deleted_count = db.cleanup_old_task_records(days)
    logger.info(f"🧹 Cleaned up {deleted_count} old tasks by {current_user.username}")
    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"Cleaned up {deleted_count} tasks older than {days} days",
    }


@router.post("/admin/reset-stale", tags=["系统管理"])
async def reset_stale_tasks(
    timeout_minutes: int = Query(60, description="超时时间（分钟）"),
    current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE)),
):
    """
    重置超时的 processing 任务（管理接口）
    """
    reset_count = db.reset_stale_tasks(timeout_minutes)
    logger.info(f"🔄 Reset {reset_count} stale tasks by {current_user.username}")
    return {
        "success": True,
        "reset_count": reset_count,
        "message": f"Reset tasks processing for more than {timeout_minutes} minutes",
    }


@router.get("/engines", tags=["系统信息"])
async def list_engines():
    """
    列出所有可用的处理引擎
    """
    engines = {
        "document": [
            {
                "name": "pipeline",
                "display_name": "Standard Pipeline (通用管道)",
                "description": "基于 PDF-Extract-Kit 的传统多模型管道，速度快，无幻觉，适合大多数文档。",
                "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"],
            },
            {
                "name": "vlm-auto-engine",
                "display_name": "MinerU 2.5 VLM (视觉大模型)",
                "description": "基于 MinerU 2.5 (1.2B) 视觉模型，擅长处理复杂排版、图表和非标准文档。",
                "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"],
            },
            {
                "name": "hybrid-auto-engine",
                "display_name": "Hybrid High-Precision (高精度混合)",
                "description": "结合 Pipeline 的稳定性与 VLM 的理解能力，提供最高精度的解析效果。",
                "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"],
            },
        ],
        "ocr": [],
        "audio": [],
        "video": [],
        "format": [],
        "office": [
            {
                "name": "MarkItDown (快速)",
                "value": "auto",
                "description": "Office 文档和文本文件转换引擎（快速但图片提取可能不完整）",
                "supported_formats": [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".html", ".txt", ".csv"],
            },
            {
                "name": "LibreOffice + MinerU (完整)",
                "value": "auto",
                "description": "将 Office 文件转为 PDF 后使用 MinerU 处理（慢但图片提取完整）",
                "supported_formats": [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"],
            }
        ],
    }

    import importlib.util

    if importlib.util.find_spec("paddleocr_vl") is not None:
        engines["ocr"].append({"name": "paddleocr_vl", "display_name": "PaddleOCR-VL v1.5 (0.9B)", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]})

    if importlib.util.find_spec("paddleocr_vl_vllm") is not None:
        engines["ocr"].append({"name": "paddleocr-vl-vllm", "display_name": "PaddleOCR-VL v1.5 (0.9B) (vLLM)", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]})

    if importlib.util.find_spec("audio_engines") is not None:
        engines["audio"].append({"name": "sensevoice", "display_name": "SenseVoice", "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"]})

    if importlib.util.find_spec("video_engines") is not None:
        engines["video"].append({"name": "video", "display_name": "Video Processing", "supported_formats": [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]})

    try:
        from format_engines import FormatEngineRegistry
        for engine_info in FormatEngineRegistry.list_engines():
            engines["format"].append({
                "name": engine_info["name"],
                "display_name": engine_info["name"].upper(),
                "description": engine_info["description"],
                "supported_formats": engine_info["extensions"],
            })
    except ImportError:
        pass

    return {
        "success": True,
        "engines": engines,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health", tags=["系统信息"])
async def health_check():
    """
    健康检查接口
    """
    try:
        stats = db.get_queue_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "queue_stats": stats,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


# ============================================================================
# 自定义文件服务（统一接口，支持 URL 编码与 MIME 识别）
# ============================================================================
@router.get("/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    """提供输出文件的访问服务"""
    try:
        # 解码并移除开头的斜杠，防止 double slash 或 encoding 问题
        decoded_path = unquote(file_path).lstrip("/")
        
        # 拼接完整路径
        full_path = (OUTPUT_DIR / decoded_path).resolve()
        
        logger.debug(f"📥 Serving output file: {full_path}")

        # 防止目录穿越
        if not full_path.is_relative_to(OUTPUT_DIR.resolve()) or not full_path.is_file():
            logger.warning(f"❌ Access denied or file not found: {full_path}")
            raise HTTPException(status_code=404, detail="File not found or access denied")

        # 自动猜测 MIME 类型
        media_type, _ = mimetypes.guess_type(full_path)
        media_type = media_type or "application/octet-stream"

        # ✅ [修复 Bug 4] 强制浏览器内联预览 (inline)，不使用 filename 参数以免触发 attachment 下载
        headers = {
            "Content-Disposition": f"inline; filename*=utf-8''{quote(full_path.name)}"
        }
        
        return FileResponse(
            path=str(full_path), 
            media_type=media_type, 
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving output file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/files/upload/{file_path:path}", tags=["文件服务"])
async def serve_upload_file(file_path: str):
    """提供上传源文件的访问服务"""
    try:
        # 解码并移除开头的斜杠
        decoded_path = unquote(file_path).lstrip("/")
        
        # 拼接完整路径
        full_path = (UPLOAD_DIR / decoded_path).resolve()
        
        logger.debug(f"📥 Serving upload file: {full_path}")

        # 防止目录穿越
        if not full_path.is_relative_to(UPLOAD_DIR.resolve()) or not full_path.is_file():
            logger.warning(f"❌ Access denied or file not found: {full_path}")
            raise HTTPException(status_code=404, detail="File not found or access denied")

        # 自动猜测 MIME 类型
        media_type, _ = mimetypes.guess_type(full_path)
        media_type = media_type or "application/octet-stream"

        # ✅ [修复 Bug 4] 强制浏览器内联预览 (inline)，不使用 filename 参数以免触发 attachment 下载
        headers = {
            "Content-Disposition": f"inline; filename*=utf-8''{quote(full_path.name)}"
        }

        return FileResponse(
            path=str(full_path), 
            media_type=media_type, 
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving upload file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# 注册双重路由
# ============================================================================
app.include_router(router, prefix="/api/v1")
app.include_router(router, prefix="/v1")


logger.info(f"📁 File service mounted: /api/v1/files/output -> {OUTPUT_DIR}")
logger.info(f"📁 File service mounted: /api/v1/files/upload -> {UPLOAD_DIR}")

if __name__ == "__main__":
    # 从环境变量读取端口，默认为8000
    api_port = int(os.getenv("API_PORT", "8000"))

    logger.info("🚀 Starting MinerU Tianshu API Server...")
    logger.info(f"📖 API Documentation: http://localhost:{api_port}/docs")

    uvicorn.run(app, host="0.0.0.0", port=api_port, log_level="info")
