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
import mimetypes
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

# ✅ [优化] 预注册 MIME 类型，防止精简环境识别失败导致浏览器强制下载
mimetypes.add_type('application/pdf', '.pdf')
mimetypes.add_type('image/png', '.png')
mimetypes.add_type('image/jpeg', '.jpg')
mimetypes.add_type('image/jpeg', '.jpeg')

# 初始化 FastAPI 应用
app = FastAPI(
    title="MinerU Tianshu API",
    description="天枢 - 企业级 AI 数据预处理平台 | 支持文档、图片、音频、视频等多模态数据处理",
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
db_path = str(Path(db_path_env).resolve()) if db_path_env else str((PROJECT_ROOT / "data" / "db" / "mineru_tianshu.db").resolve())
db = TaskDB(db_path)
auth_db = AuthDB()

# 注册认证路由
app.include_router(auth_router)

# 目录配置
output_path_env = os.getenv("OUTPUT_PATH")
OUTPUT_DIR = Path(output_path_env) if output_path_env else PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

upload_path_env = os.getenv("UPLOAD_PATH")
UPLOAD_DIR = Path(upload_path_env) if upload_path_env else PROJECT_ROOT / "input"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_markdown_images_legacy(md_content: str, image_dir: Path, result_path: str):
    """【向后兼容】处理 Markdown 中的图片本地引用"""
    if "http://" in md_content or "https://" in md_content or not image_dir.exists():
        return md_content

    def replace_image_path(match):
        full_match, alt_text, image_path = match.group(0), match.group(1), match.group(2)
        if image_path.startswith("http"): 
            return full_match
        try:
            image_filename = Path(image_path).name
            output_dir_str = str(OUTPUT_DIR).replace("\\", "/")
            result_path_str = result_path.replace("\\", "/")
            if result_path_str.startswith(output_dir_str):
                relative_path = result_path_str[len(output_dir_str) :].lstrip("/")
                # ✅ [修复Bug] 必须加 safe="/"，否则路径里的斜杠会变成 %2F 导致图片 404
                static_url = f"/api/v1/files/output/{quote(relative_path, safe='/')}/images/{quote(image_filename, safe='/')}"
                if "![" in full_match:
                    return f"![{alt_text}]({static_url})"
                else:
                    return full_match.replace(image_path, static_url)
        except Exception as e: 
            logger.warning(f"Image replacement failed: {e}")
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
        with open(temp_file_path, "wb") as f:
            while chunk := await file.read(1 << 23): f.write(chunk)

        options = {
            "lang": lang, "method": method, "formula_enable": formula_enable, "table_enable": table_enable,
            "start_page": start_page, "end_page": end_page, "force_ocr": force_ocr, "server_url": server_url,
            "draw_layout_bbox": draw_layout_bbox, "draw_span_bbox": draw_span_bbox, "dump_markdown": dump_markdown,
            "dump_middle_json": dump_middle_json, "dump_model_output": dump_model_output, "dump_content_list": dump_content_list,
            "dump_orig_pdf": dump_orig_pdf, "keep_audio": keep_audio, "enable_keyframe_ocr": enable_keyframe_ocr,
            "ocr_backend": ocr_backend, "keep_keyframes": keep_keyframes, "enable_speaker_diarization": enable_speaker_diarization,
            "remove_watermark": remove_watermark, "watermark_conf_threshold": watermark_conf_threshold,
            "watermark_dilation": watermark_dilation, "convert_office_to_pdf": convert_office_to_pdf,
        }

        task_id = db.create_task(file_name=file.filename, file_path=str(temp_file_path), backend=backend, options=options, priority=priority, user_id=current_user.user_id)
        return {"success": True, "task_id": task_id, "status": "pending", "file_name": file.filename}
    except Exception as e:
        logger.error(f"❌ Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def get_task_status(
    task_id: str,
    format: str = Query("markdown", description="返回格式: markdown/json/both"),
    current_user: User = Depends(get_current_active_user),
):
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")

    if not current_user.has_permission(Permission.TASK_VIEW_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Permission denied")

    # ✅ [确认修复] 前端请求 source_url，这里生成的带 /api/v1/
    source_url = f"/api/v1/files/upload/{quote(Path(task['file_path']).name)}" if task.get("file_path") else None
    response = {
        "success": True, "task_id": task_id, "status": task["status"], "file_name": task["file_name"],
        "source_url": source_url, "backend": task["backend"], "error_message": task["error_message"],
        "created_at": task["created_at"], "started_at": task["started_at"], "completed_at": task["completed_at"],
    }

    if task.get("is_parent"):
        child_count, child_completed = task.get("child_count", 0), task.get("child_completed", 0)
        response["is_parent"] = True
        response["subtask_progress"] = {"total": child_count, "completed": child_completed, "percentage": round(child_completed / child_count * 100, 1) if child_count > 0 else 0}

    if task["status"] == "completed" and task["result_path"]:
        res_dir = Path(task["result_path"])
        if res_dir.exists():
            response["data"] = {}
            # 查找预览PDF
            pdf = next(res_dir.rglob("*_layout.pdf"), next(res_dir.rglob("*.pdf"), None))
            if pdf: 
                try: 
                    # ✅ [修复Bug] 保证路径斜杠不被错误编码
                    response["data"]["pdf_path"] = quote(str(pdf.relative_to(OUTPUT_DIR)).replace("\\", "/"), safe="/")
                except: pass

            if format in ["markdown", "both"]:
                md = next(res_dir.rglob("result.md"), next(res_dir.rglob("*.md"), None))
                if md:
                    with open(md, "r", encoding="utf-8") as f:
                        response["data"]["content"] = process_markdown_images_legacy(f.read(), md.parent / "images", task["result_path"])
            
            if format in ["json", "both"]:
                js = next((f for f in res_dir.rglob("*.json") if "_content_list.json" in f.name or f.name == "result.json"), None)
                if js:
                    with open(js, "r", encoding="utf-8") as f:
                        response["data"]["json_content"] = json.load(f)
    return response


@app.delete("/api/v1/tasks/{task_id}", tags=["任务管理"])
async def cancel_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    task = db.get_task(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")

    if not current_user.has_permission(Permission.TASK_DELETE_ALL) and task.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Permission denied")

    if task["status"] == "pending":
        db.update_task_status(task_id, "cancelled")
        file_path = Path(task["file_path"])
        if file_path.exists(): file_path.unlink()
        return {"success": True, "message": "Task cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in {task['status']} status")


@app.get("/api/v1/queue/tasks", tags=["队列管理"])
async def list_tasks(
    status: Optional[str] = Query(None, description="筛选状态"),
    backend: Optional[str] = Query(None, description="筛选后端引擎"),
    search: Optional[str] = Query(None, description="搜索文件名或任务ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    current_user: User = Depends(get_current_active_user),
):
    can_view_all = current_user.has_permission(Permission.TASK_VIEW_ALL)
    conds, params = [], []

    if not can_view_all:
        conds.append("user_id = ?"), params.append(current_user.user_id)
    if status:
        conds.append("status = ?"), params.append(status)
    if backend:
        conds.append("backend = ?"), params.append(backend)
    if search:
        search_term = f"%{search.strip()}%"
        conds.append("(file_name LIKE ? OR task_id = ?)"), params.extend([search_term, search.strip()])

    where = " WHERE " + " AND ".join(conds) if conds else ""
    offset = (page - 1) * page_size

    with db.get_cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM tasks{where}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [page_size, offset])
        tasks = [dict(row) for row in cursor.fetchall()]

    return {"success": True, "total": total, "page": page, "page_size": page_size, "tasks": tasks, "can_view_all": can_view_all}


# ✅ [修复Bug] 恢复您丢失的管理接口
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
    import importlib.util
    ocr = []
    if importlib.util.find_spec("paddleocr_vl"):
        ocr.append({"name": "paddleocr_vl", "display_name": "PaddleOCR-VL v1.5 (0.9B)"})
    if importlib.util.find_spec("paddleocr_vl_vllm"):
        ocr.append({"name": "paddleocr-vl-vllm", "display_name": "PaddleOCR-VL v1.5 (0.9B) (vLLM)"}) 
    
    return {
        "success": True,
        "engines": {
            "document": [
                {"name": "pipeline", "display_name": "Standard Pipeline"},
                {"name": "vlm-auto-engine", "display_name": "MinerU 2.5 VLM"},
                {"name": "hybrid-auto-engine", "display_name": "Hybrid High-Precision"}
            ],
            "ocr": ocr,
            "audio": [{"name": "sensevoice", "display_name": "SenseVoice"}],
            "video": [{"name": "video", "display_name": "Video Processing"}]
        }
    }


@app.get("/api/v1/health", tags=["系统信息"])
async def health_check():
    try:
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "database": "connected", "queue_stats": db.get_queue_stats()}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


# ✅ [核心修复] 将文件路由前缀补齐 /api ，解决 404 错误
# ✅ [核心修复] 使用 Pathlib 的 is_relative_to 修复安全漏洞
@app.get("/api/v1/files/{file_type}/{file_path:path}", tags=["文件服务"])
async def serve_file(file_type: str, file_path: str):
    if file_type not in ["output", "upload"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    root_dir = OUTPUT_DIR.resolve() if file_type == "output" else UPLOAD_DIR.resolve()
    
    try:
        decoded_path = unquote(file_path)
        full_path = (root_dir / decoded_path).resolve()
        
        # 安全防御：防止目录穿越攻击
        if not full_path.is_relative_to(root_dir) or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        mtype, _ = mimetypes.guess_type(full_path)
        return FileResponse(path=str(full_path), media_type=mtype or "application/octet-stream", filename=full_path.name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving {file_type} file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    api_port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"🚀 MinerU Tianshu API Server starting on port {api_port}...")
    uvicorn.run(app, host="0.0.0.0", port=api_port)
