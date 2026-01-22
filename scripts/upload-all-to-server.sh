#!/bin/bash
# Tianshu 上传脚本
# 只上传必要的文件到服务器，然后自动部署

set -e

# ============================================================================
# 配置
# ============================================================================
SERVER_USER="${1:-serverName}"
SERVER_HOST="${2:-100.200.300.400}"
SERVER_PATH="${3:-~/serverDir}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(dirname "${SCRIPT_DIR}")/docker-images"
TEMP_DIR="${SCRIPT_DIR}/.upload_temp"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 使用说明
# ============================================================================
show_usage() {
    echo "Usage: $0 [server_user] [server_host] [server_path]"
    echo ""
    echo "Arguments:"
    echo "  server_user   Server username (default: gpu)"
    echo "  server_host   Server hostname or IP (default: 192.168.100.27)"
    echo "  server_path   Remote directory path (default: ~/tianshu-mineru)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 root 192.168.1.100 /opt/tianshu"
    echo ""
}

# ============================================================================
# 清理本地目录
# ============================================================================
cleanup_local() {
    log_info "清理本地目录..."

    # 删除已解压的 models-offline/ 目录
    if [ -d "${LOCAL_DIR}/models-offline" ]; then
        log_info "  删除 models-offline/ 目录..."
        rm -rf "${LOCAL_DIR}/models-offline"
    fi

    # 删除 models/ 目录
    if [ -d "${LOCAL_DIR}/models" ]; then
        log_info "  删除 models/ 目录..."
        rm -rf "${LOCAL_DIR}/models"
    fi

    # 删除 data/ 目录
    if [ -d "${LOCAL_DIR}/data" ]; then
        log_info "  删除 data/ 目录..."
        rm -rf "${LOCAL_DIR}/data"
    fi

    # 删除 logs/ 目录
    if [ -d "${LOCAL_DIR}/logs" ]; then
        log_info "  删除 logs/ 目录..."
        rm -rf "${LOCAL_DIR}/logs"
    fi

    log_success "本地目录清理完成"
}

# ============================================================================
# 准备上传文件
# ============================================================================
prepare_upload() {
    log_info "准备上传文件..."

    # 创建临时目录
    mkdir -p "${TEMP_DIR}"

    # 复制必要的文件到临时目录
    log_info "  复制配置文件..."
    cp -r "${LOCAL_DIR}/.env.example" "${TEMP_DIR}/" 2>/dev/null || true
    cp -r "${LOCAL_DIR}/docker-compose.yml" "${TEMP_DIR}/" 2>/dev/null || true
    cp -r "${LOCAL_DIR}/docker-compose.offline.yml" "${TEMP_DIR}/" 2>/dev/null || true
    cp -r "${LOCAL_DIR}/deploy-offline.sh" "${TEMP_DIR}/" 2>/dev/null || true
    cp -r "${LOCAL_DIR}/mcp_config.example.json" "${TEMP_DIR}/" 2>/dev/null || true

    # 复制 Docker 镜像文件
    log_info "  复制 Docker 镜像文件..."
    if [ -f "${LOCAL_DIR}/models-offline.tar.gz" ]; then
        cp "${LOCAL_DIR}/models-offline.tar.gz" "${TEMP_DIR}/"
        log_info "    - models-offline.tar.gz"
    else
        log_error "models-offline.tar.gz 不存在！"
        exit 1
    fi

    if [ -f "${LOCAL_DIR}/rustfs-amd64.tar.gz" ]; then
        cp "${LOCAL_DIR}/rustfs-amd64.tar.gz" "${TEMP_DIR}/"
        log_info "    - rustfs-amd64.tar.gz"
    else
        log_error "rustfs-amd64.tar.gz 不存在！"
        exit 1
    fi

    if [ -f "${LOCAL_DIR}/tianshu-backend-amd64.tar.gz" ]; then
        cp "${LOCAL_DIR}/tianshu-backend-amd64.tar.gz" "${TEMP_DIR}/"
        log_info "    - tianshu-backend-amd64.tar.gz"
    else
        log_error "tianshu-backend-amd64.tar.gz 不存在！"
        exit 1
    fi

    if [ -f "${LOCAL_DIR}/tianshu-frontend-amd64.tar.gz" ]; then
        cp "${LOCAL_DIR}/tianshu-frontend-amd64.tar.gz" "${TEMP_DIR}/"
        log_info "    - tianshu-frontend-amd64.tar.gz"
    else
        log_error "tianshu-frontend-amd64.tar.gz 不存在！"
        exit 1
    fi

    log_success "文件准备完成"
}

# ============================================================================
# 上传到服务器
# ============================================================================
upload_to_server() {
    log_info "上传文件到服务器..."
    log_info "  服务器: ${SERVER_USER}@${SERVER_HOST}"
    log_info "  路径: ${SERVER_PATH}"
    echo ""

    # 使用 rsync 上传
    rsync -avz --progress "${TEMP_DIR}/" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"

    log_success "上传完成"
}

# ============================================================================
# 清理临时目录
# ============================================================================
cleanup_temp() {
    log_info "清理临时目录..."
    rm -rf "${TEMP_DIR}"
    log_success "临时目录清理完成"
}

# ============================================================================
# 在服务器上部署
# ============================================================================
deploy_on_server() {
    log_info "在服务器上部署..."

    # 检查是否需要自动部署
    read -p "是否在服务器上自动部署？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "跳过自动部署"
        return
    fi

    # 在服务器上执行部署脚本
    ssh "${SERVER_USER}@${SERVER_HOST}" "cd ${SERVER_PATH} && bash deploy-offline.sh"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    log_info "=========================================="
    log_info "📦 Tianshu 上传脚本"
    log_info "=========================================="
    echo ""

    # 显示配置
    log_info "配置信息:"
    echo "  本地目录: ${LOCAL_DIR}"
    echo "  服务器: ${SERVER_USER}@${SERVER_HOST}"
    echo "  远程路径: ${SERVER_PATH}"
    echo ""

    # 清理本地目录
    cleanup_local
    echo ""

    # 准备上传文件
    prepare_upload
    echo ""

    # 上传到服务器
    upload_to_server
    echo ""

    # 清理临时目录
    cleanup_temp
    echo ""

    # 在服务器上部署
    deploy_on_server
    echo ""

    log_info "=========================================="
    log_success "✅ 上传完成！"
    log_info "=========================================="
    echo ""

    log_info "后续步骤:"
    echo "  1. 登录服务器: ssh ${SERVER_USER}@${SERVER_HOST}"
    echo "  2. 进入目录: cd ${SERVER_PATH}"
    echo "  3. 查看状态: docker compose ps"
    echo "  4. 查看日志: docker compose logs -f"
}

# 捕获中断信号
trap 'log_warning "操作被用户中断"; cleanup_temp; exit 130' SIGINT SIGTERM

# 执行主函数
main
