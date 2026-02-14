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
import mimetypes  # ✅ 修复：用于识别文件类型以支持预览
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
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

# 初始化 FastAPI 应用
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持文档、图片、音频、视频等多模态数据处理 | 企业级认证授权",
    version="2.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 初始化数据库
db_path_env = os.getenv("DATABASE_PATH")
if db_path_env:
    db_path = str(Path(db_path_env).resolve())
    logger.info(f"📊 API Server using DATABASE_PATH: {db_path_env} -> {db_path}")
    db = TaskDB(db_path)
else:
    logger.warning("⚠️  DATABASE_PATH not set in API Server, using default")
    default_db_path = PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db"
    default_db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = str(default_db_path.resolve())
    logger.info(f"📊 Using default database path: {db_path}")
    db = TaskDB(db_path)
auth_db = AuthDB()

# 注册认证路由
app.include_router(auth_router)

# ==============================================================================
# 目录配置
# ==============================================================================
output_path_env = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(output_path_env) if output_path_env else PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

upload_path_env = os.getenv("UPLOAD_PATH")
UPLOAD_DIR = Path(upload_path_env) if upload_path_env else PROJECT_ROOT / "input"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# 注意：此函数已废弃，Worker 已自动上传图片到 RustFS 并替换 URL
def process_markdown_images_legacy(md_content: str, image_dir: Path, result_path: str):
    if "http://" in md_content or "https://" in md_content:
        return md_content
    if not image_dir.exists():
        return md_content

    def replace_image_path(match):
        full_match = match.group(0)
        image_path = match.group(2)
        alt_text = match.group(1) if "![" in full_match else "Image"
        if image_path.startswith("http"):
            return full_match
        try:
            image_filename = Path(image_path).name
            output_dir_str = str(OUTPUT_DIR).replace("\\", "/")
            result_path_str = result_path.replace("\\", "/")
            if result_path_str.startswith(output_dir_str):
                relative_path = result_path_str[len(output_dir_str) :].lstrip("/")
                encoded_relative_path = quote(relative_path, safe="/")
                encoded_filename = quote(image_filename, safe="/")
                static_url = f"/api/v1/files/output/{encoded_relative_path}/images/{encoded_filename}"
                return f"![{alt_text}]({static_url})" if "![" in full_match else full_match.replace(image_path, static_url)
        except: pass
        return full_match

    md_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    html_pattern = r'<img\s+([^>]*\s+)?src="([^"]+)"([^>]*)>'
    new_content = re.sub(md_pattern, replace_image_path, md_content)
    new_content = re.sub(html_pattern, replace_image_path, new_content)
    return new_content

@app.get("/", tags=["系统信息"])
async def root():
    return {"service": "MinerU Tianshu", "version": "2.0.0", "docs": "/docs"}

@app.post("/api/v1/tasks/submit", tags=["任务管理"])
async def submit_task(
    file: UploadFile = File(..., description="文件: PDF/图片/Office/音频/视频等"),
    backend: str = Form("auto"),
    lang: str = Form("auto"),
    method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    priority: int = Form(0),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    force_ocr: bool = Form(False),
    server_url: Optional[str] = Form(None),
    draw_layout_bbox: bool = Form(True),
    draw_span_bbox: bool = Form(True),
    dump_markdown: bool = Form(True),
    dump_middle_json: bool = Form(True),
    dump_model_output: bool = Form(True),
    dump_content_list: bool = Form(True),
    dump_orig_pdf: bool = Form(True),
    draw_layout: bool = Form(True),
    draw_span: bool = Form(True),
    keep_audio: bool = Form(False),
    enable_keyframe_ocr: bool = Form(False),
    ocr_backend: str = Form("paddleocr-vl"),
    keep_keyframes: bool = Form(False),
    enable_speaker_diarization: bool = Form(False),
    remove_watermark: bool = Form(False),
    watermark_conf_threshold: float = Form(0.35),
    watermark_dilation: int = Form(10),
    convert_office_to_pdf: bool = Form(False),
    current_user: User = Depends(require_permission(Permission.TASK_SUBMIT)),
):
    try:
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        temp_file_path = UPLOAD_DIR / unique_filename
        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(1 << 23)
                if not chunk: break
                temp_file.write(chunk)
        
        options = {
            "lang": lang, "method": method, "formula_enable": formula_enable, "table_enable": table_enable,
            "start_page": start_page, "end_page": end_page, "force_ocr": force_ocr, "server_url": server_url,
            "draw_layout_bbox": draw_layout_bbox, "draw_span_bbox": draw_span_bbox, "dump_markdown": dump_markdown,
            "dump_middle_json": dump_middle_json, "dump_model_output": dump_model_output, "dump_content_list": dump_content_list,
            "dump_orig_pdf": dump_orig_pdf, "draw_layout": draw_layout, "draw_span": draw_span,
            "keep_audio": keep_audio, "enable_keyframe_ocr": enable_keyframe_ocr, "ocr_backend": ocr_backend,
            "keep_keyframes": keep_keyframes, "enable_speaker_diarization": enable_speaker_diarization,
            "remove_watermark": remove_watermark, "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation, "convert_office_to_pdf": convert_office_to_pdf
        }
        task_id = db.create_task(file_name=file.filename, file_path=str(temp_file_path), backend=backend, options=options, priority=priority, user_id=current_user.user_id)
        return {"success": True, "task_id": task_id, "status": "pending", "file_name": file.filename}
    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str,
    upload_images: bool = Query(False),
    format: str = Query("markdown", description="markdown/json/both"),
    current_user: User = Depends(get_current_active_user),
):
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")
    if not current_user.has_permission(Permission.TASK_VIEW_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Permission denied")

    source_url = f"/api/v1/files/upload/{quote(Path(task['file_path']).name)}" if task.get("file_path") else None
    response = {
        "success": True, "task_id": task_id, "status": task["status"], "file_name": task["file_name"],
        "source_url": source_url, "backend": task["backend"], "priority": task["priority"],
        "error_message": task["error_message"], "created_at": task["created_at"],
        "started_at": task["started_at"], "completed_at": task["completed_at"], "user_id": task.get("user_id"),
    }

    if task.get("is_parent"):
        child_count, child_completed = task.get("child_count", 0), task.get("child_completed", 0)
        response["is_parent"] = True
        response["subtask_progress"] = {"total": child_count, "completed": child_completed, "percentage": round(child_completed / child_count * 100, 1) if child_count > 0 else 0}

    if task["status"] == "completed" and task["result_path"]:
        result_dir = Path(task["result_path"])
        if result_dir.exists():
            response["data"] = {}
            md_files = list(result_dir.rglob("*.md"))
            json_files = [f for f in result_dir.rglob("*.json") if not f.parent.name.startswith("page_") and (f.name in ["content.json", "result.json"] or "_content_list.json" in f.name)]
            
            pdf_files = list(result_dir.rglob("*.pdf"))
            preview_pdf = next((p for p in pdf_files if "_layout.pdf" in p.name), next((p for p in pdf_files if "_span.pdf" in p.name), None))
            if preview_pdf:
                try: response["data"]["pdf_path"] = quote(str(preview_pdf.relative_to(OUTPUT_DIR)).replace("\\", "/"))
                except: pass

            if format in ["markdown", "both"] and md_files:
                md_file = next((f for f in md_files if f.name == "result.md"), md_files[0])
                with open(md_file, "r", encoding="utf-8") as f:
                    response["data"]["content"] = process_markdown_images_legacy(f.read(), md_file.parent / "images", task["result_path"])
            
            if format in ["json", "both"] and json_files:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    response["data"]["json_content"] = json.load(f)
    return response

@app.get("/api/v1/queue/tasks", tags=["队列管理"])
async def list_tasks(
    status: Optional[str] = Query(None),
    backend: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    can_view_all = current_user.has_permission(Permission.TASK_VIEW_ALL)
    conditions, params = [], []
    if not can_view_all:
        conditions.append("user_id = ?"), params.append(current_user.user_id)
    if status:
        conditions.append("status = ?"), params.append(status)
    if backend:
        conditions.append("backend = ?"), params.append(backend)
    if search:
        conditions.append("(file_name LIKE ? OR task_id = ?)"), params.append(f"%{search.strip()}%"), params.append(search.strip())

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    offset = (page - 1) * page_size
    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM tasks{where_clause}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT * FROM tasks {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [page_size, offset])
        tasks = [dict(row) for row in cursor.fetchall()]
    return {"success": True, "total": total, "page": page, "page_size": page_size, "tasks": tasks, "can_view_all": can_view_all}

@app.get("/api/v1/queue/stats", tags=["队列管理"])
async def get_queue_stats(current_user: User = Depends(require_permission(Permission.QUEUE_VIEW))):
    stats = db.get_queue_stats()
    return {"success": True, "stats": stats, "total": sum(stats.values()), "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/admin/cleanup", tags=["系统管理"])
async def cleanup_old_tasks(days: int = Query(7), current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE))):
    deleted_count = db.cleanup_old_task_records(days)
    return {"success": True, "deleted_count": deleted_count}

@app.post("/api/v1/admin/reset-stale", tags=["系统管理"])
async def reset_stale_tasks(timeout_minutes: int = Query(60), current_user: User = Depends(require_permission(Permission.QUEUE_MANAGE))):
    reset_count = db.reset_stale_tasks(timeout_minutes)
    return {"success": True, "reset_count": reset_count}

@app.get("/api/v1/engines", tags=["系统信息"])
async def list_engines():
    engines = {
        "document": [
            {"name": "pipeline", "display_name": "Standard Pipeline", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]},
            {"name": "vlm-auto-engine", "display_name": "MinerU 2.5 VLM", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]},
            {"name": "hybrid-auto-engine", "display_name": "Hybrid High-Precision", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]}
        ],
        "ocr": [], "audio": [], "video": [], "format": [],
        "office": [
            {"name": "MarkItDown (快速)", "value": "auto", "supported_formats": [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".html", ".txt", ".csv"]},
            {"name": "LibreOffice + MinerU (完整)", "value": "auto", "supported_formats": [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"]}
        ]
    }
    import importlib.util
    if importlib.util.find_spec("paddleocr_vl"):
        engines["ocr"].append({"name": "paddleocr_vl", "display_name": "PaddleOCR-VL v1.5 (0.9B)", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]})
    if importlib.util.find_spec("paddleocr_vl_vllm"):
        engines["ocr"].append({"name": "paddleocr-vl-vllm", "display_name": "PaddleOCR-VL v1.5 (0.9B) (vLLM)", "supported_formats": [".pdf", ".png", ".jpg", ".jpeg"]})
    if importlib.util.find_spec("audio_engines"):
        engines["audio"].append({"name": "sensevoice", "display_name": "SenseVoice", "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"]})
    if importlib.util.find_spec("video_engines"):
        engines["video"].append({"name": "video", "display_name": "Video Processing", "supported_formats": [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]})
    return {"success": True, "engines": engines}

@app.get("/api/v1/health", tags=["系统信息"])
async def health_check():
    try:
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "database": "connected", "queue_stats": db.get_queue_stats()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/v1/files/output/{file_path:path}", tags=["文件服务"])
async def serve_output_file(file_path: str):
    try:
        full_path = (OUTPUT_DIR / unquote(file_path)).resolve()
        if not str(full_path).startswith(str(OUTPUT_DIR.resolve())) or not full_path.is_file():
            raise HTTPException(status_code=404)
        media_type, _ = mimetypes.guess_type(full_path)
        return FileResponse(path=str(full_path), media_type=media_type or "application/octet-stream", filename=full_path.name)
    except: raise HTTPException(status_code=500)

@app.get("/v1/files/upload/{file_path:path}", tags=["文件服务"])
async def serve_upload_file(file_path: str):
    try:
        full_path = (UPLOAD_DIR / unquote(file_path)).resolve()
        if not str(full_path).startswith(str(UPLOAD_DIR.resolve())) or not full_path.is_file():
            raise HTTPException(status_code=404)
        media_type, _ = mimetypes.guess_type(full_path)
        return FileResponse(path=str(full_path), media_type=media_type or "application/octet-stream", filename=full_path.name)
    except: raise HTTPException(status_code=500)

if __name__ == "__main__":
    api_port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"🚀 Starting Tianshu API Server on port {api_port}...")
    uvicorn.run(app, host="0.0.0.0", port=api_port)
