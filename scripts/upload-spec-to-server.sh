#!/bin/bash
# Tianshu 指定文件上传脚本
# 用于增量更新，只上传指定的文件到服务器

set -e

# ============================================================================
# 配置
# ============================================================================
SERVER_USER="${1}"
SERVER_HOST="${2}"
SERVER_PATH="${3}"
FILE_SPEC="${4}"  # 指定要上传的文件: backend, frontend, rustfs, models, config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(dirname "${SCRIPT_DIR}")/docker-images"

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
    echo "Usage: $0 <server_user> <server_host> <server_path> <file_spec>"
    echo ""
    echo "Arguments:"
    echo "  server_user   Server username (required)"
    echo "  server_host   Server hostname or IP (required)"
    echo "  server_path   Remote directory path (required)"
    echo "  file_spec     File specification (required)"
    echo ""
    echo "File Specification:"
    echo "  backend       后端镜像 (tianshu-backend-amd64.tar.gz)"
    echo "  frontend      前端镜像 (tianshu-frontend-amd64.tar.gz)"
    echo "  rustfs        RustFS 镜像 (rustfs-amd64.tar.gz)"
    echo "  models        模型文件 (models-offline.tar.gz)"
    echo "  config        配置文件 (.env.example, docker-compose.yml 等)"
    echo ""
    echo "Examples:"
    echo "  # 只上传后端镜像"
    echo "  $0 root 192.168.1.100 /opt/tianshu backend"
    echo ""
    echo "  # 只上传前端镜像"
    echo "  $0 root 192.168.1.100 /opt/tianshu frontend"
    echo ""
    echo "  # 只上传配置文件"
    echo "  $0 root 192.168.1.100 /opt/tianshu config"
    echo ""
    echo "  # 只上传模型文件"
    echo "  $0 root 192.168.1.100 /opt/tianshu models"
    echo ""
}

# ============================================================================
# 检查参数
# ============================================================================
check_arguments() {
    if [ -z "$SERVER_USER" ] || [ -z "$SERVER_HOST" ] || [ -z "$SERVER_PATH" ] || [ -z "$FILE_SPEC" ]; then
        log_error "缺少必需参数！"
        echo ""
        show_usage
        exit 1
    fi

    # 验证 file_spec
    case "$FILE_SPEC" in
        backend|frontend|rustfs|models|config)
            ;;
        *)
            log_error "无效的文件规格: $FILE_SPEC"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# ============================================================================
# 获取文件信息
# ============================================================================
get_file_info() {
    local file_spec=$1

    case "$file_spec" in
        backend)
            echo "tianshu-backend-amd64.tar.gz"
            ;;
        frontend)
            echo "tianshu-frontend-amd64.tar.gz"
            ;;
        rustfs)
            echo "rustfs-amd64.tar.gz"
            ;;
        models)
            echo "models-offline.tar.gz"
            ;;
        config)
            echo "config_files"  # 特殊标记
            ;;
    esac
}

# ============================================================================
# 上传文件
# ============================================================================
upload_file() {
    local file_spec=$1
    local file_info=$(get_file_info "$file_spec")

    if [ "$file_info" = "config_files" ]; then
        # 上传配置文件
        log_info "上传配置文件..."

        local config_files=""
        [ -f "${LOCAL_DIR}/.env.example" ] && config_files="${config_files} ${LOCAL_DIR}/.env.example"
        [ -f "${LOCAL_DIR}/docker-compose.yml" ] && config_files="${config_files} ${LOCAL_DIR}/docker-compose.yml"
        [ -f "${LOCAL_DIR}/docker-compose.offline.yml" ] && config_files="${config_files} ${LOCAL_DIR}/docker-compose.offline.yml"
        [ -f "${LOCAL_DIR}/deploy-offline.sh" ] && config_files="${config_files} ${LOCAL_DIR}/deploy-offline.sh"
        [ -d "${LOCAL_DIR}/mcp_config.example.json" ] && config_files="${config_files} ${LOCAL_DIR}/mcp_config.example.json"

        if [ -z "$config_files" ]; then
            log_error "没有找到配置文件！"
            exit 1
        fi

        log_info "使用 rsync 上传配置文件..."
        rsync -avz --progress ${config_files} "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"

    else
        # 上传单个镜像文件
        local file_path="${LOCAL_DIR}/${file_info}"

        if [ ! -f "$file_path" ]; then
            log_error "文件不存在: $file_path"
            exit 1
        fi

        log_info "文件信息:"
        ls -lh "$file_path"
        echo ""

        log_info "使用 rsync 上传文件..."
        rsync -avz --progress "$file_path" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"
    fi

    log_success "上传完成"
}

# ============================================================================
# 在服务器上操作
# ============================================================================
server_operations() {
    local file_spec=$1

    # 如果是镜像文件，询问是否加载并重启
    case "$file_spec" in
        backend|frontend|rustfs)
            echo ""
            read -p "是否在服务器上重新加载镜像？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                local file_info=$(get_file_info "$file_spec")
                log_info "♻️  加载镜像..."

                ssh "${SERVER_USER}@${SERVER_HOST}" << EOF
                    cd ${SERVER_PATH}
                    docker load < ${file_info}
                    echo "✓ 镜像加载完成"
EOF
                log_success "镜像已加载"
                echo ""

                # 询问是否重启服务
                read -p "是否重启相关服务？(y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    log_info "♻️  重启服务..."

                    local services=""
                    case "$file_spec" in
                        backend)
                            services="backend worker"
                            ;;
                        frontend)
                            services="frontend"
                            ;;
                        rustfs)
                            services="rustfs"
                            ;;
                    esac

                    ssh "${SERVER_USER}@${SERVER_HOST}" << EOF
                        cd ${SERVER_PATH}
                        docker-compose restart ${services}
                        echo "✓ 服务重启完成"
EOF
                    log_success "服务已重启"
                else
                    log_info "跳过重启，请手动执行："
                    echo "  ssh ${SERVER_USER}@${SERVER_HOST}"
                    echo "  cd ${SERVER_PATH}"
                    echo "  docker-compose restart ${services}"
                fi
            else
                log_info "跳过加载，请手动执行："
                echo "  ssh ${SERVER_USER}@${SERVER_HOST}"
                echo "  cd ${SERVER_PATH}"
                echo "  docker load < $(get_file_info "$file_spec")"
                echo "  docker-compose restart ..."
            fi
            ;;
        models)
            echo ""
            log_warning "模型文件已上传，需要重新部署才能生效："
            echo "  1. 删除现有的 models-offline 目录"
            echo "  2. 解压新的 models-offline.tar.gz"
            echo "  3. 删除 Docker 卷中的 .models_initialized 标记"
            echo "  4. 重启 worker 服务"
            echo ""
            read -p "是否执行上述操作？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "♻️  更新模型..."
                ssh "${SERVER_USER}@${SERVER_HOST}" << EOF
                    cd ${SERVER_PATH}
                    rm -rf models-offline
                    tar xzf models-offline.tar.gz
                    docker-compose exec -T worker rm -f /root/.cache/.models_initialized
                    docker-compose restart worker
                    echo "✓ 模型更新完成"
EOF
                log_success "模型已更新"
            fi
            ;;
        config)
            echo ""
            log_info "配置文件已上传"
            log_warning "如果修改了 .env 或 docker-compose.yml，需要重新启动服务："
            echo "  ssh ${SERVER_USER}@${SERVER_HOST}"
            echo "  cd ${SERVER_PATH}"
            echo "  docker-compose down"
            echo "  docker-compose up -d"
            ;;
    esac
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    log_info "=========================================="
    log_info "📤 Uploading Specific File to Server"
    log_info "=========================================="
    echo ""

    # 检查参数
    check_arguments

    # 显示配置
    log_info "配置信息:"
    echo "  本地目录: ${LOCAL_DIR}"
    echo "  服务器: ${SERVER_USER}@${SERVER_HOST}"
    echo "  远程路径: ${SERVER_PATH}"
    echo "  文件类型: ${FILE_SPEC}"
    echo ""

    # 上传文件
    upload_file "$FILE_SPEC"
    echo ""

    # 服务器操作
    server_operations "$FILE_SPEC"
    echo ""

    # 完成
    log_info "=========================================="
    log_success "✅ Upload Complete!"
    log_info "=========================================="
    echo ""

    log_info "📋 查看服务状态:"
    echo "  ssh ${SERVER_USER}@${SERVER_HOST} 'cd ${SERVER_PATH} && docker-compose ps'"
    echo ""

    log_info "📋 查看服务日志:"
    echo "  ssh ${SERVER_USER}@${SERVER_HOST} 'cd ${SERVER_PATH} && docker-compose logs -f ${FILE_SPEC}'"
    echo ""
}

# 捕获中断信号
trap 'log_warning "上传被用户中断"; exit 130' SIGINT SIGTERM

# 检查帮助参数
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 执行主函数
main
