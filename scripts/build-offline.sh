#!/bin/bash
# Tianshu 离线镜像构建脚本
# 构建 GPU 版本镜像
# 用于在联网环境中构建镜像并复用已下载的模型

set -e

# ============================================================================
# 配置
# ============================================================================
PLATFORM="${PLATFORM:-amd64}"
OUTPUT_DIR="./docker-images"
MODELS_DIR="./models-offline"

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
# 主函数
# ============================================================================
main() {
    log_info "=========================================="
    log_info "🚀 Building Tianshu Image for Offline Deployment"
    log_info "=========================================="
    log_info "Platform: linux/$PLATFORM"
    log_info "Output: $OUTPUT_DIR"
    echo ""

    # 1. 检查 Docker 环境
    log_info "📋 Checking Docker environment..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_warning "docker-compose not found, trying docker compose..."
        if ! docker compose version &> /dev/null; then
            log_error "Docker Compose is not installed!"
            exit 1
        fi
    fi

    DOCKER_VERSION=$(docker --version)
    log_success "Docker: $DOCKER_VERSION"
    echo ""

    # 2. 检查 NVIDIA 环境（可选）
    log_info "🔍 Checking NVIDIA environment..."
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        NVIDIA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        log_success "NVIDIA Driver: $NVIDIA_VERSION"
        log_info "Building with GPU support"
    else
        log_warning "NVIDIA GPU not detected"
        log_info "Building images without runtime GPU access"
        log_info "Images will still support GPU when deployed on GPU servers"
    fi
    echo ""

    # 3. 检查模型是否已下载
    log_info "📦 Checking models..."
    if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
        log_warning "Models directory not found or empty!"
        log_info "Running model download script..."
        echo ""

        # 检查 Python 和依赖
        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 is not installed!"
            exit 1
        fi

        # 安装必要的依赖（如果缺失）
        python3 -m pip install --quiet huggingface-hub modelscope loguru 2>/dev/null || true

        # 执行下载
        python3 backend/download_models.py --output "$MODELS_DIR"

        if [ $? -eq 0 ]; then
            log_success "Models downloaded successfully"
        else
            log_error "Model download failed!"
            log_info "Please run manually: python3 backend/download_models.py --output $MODELS_DIR"
            exit 1
        fi
    else
        log_success "Models directory found: $MODELS_DIR"
        log_info "Skipping model download (already exists)"
        log_info "Use --force to re-download: python3 backend/download_models.py --output $MODELS_DIR --force"
    fi
    echo ""

    # 4. 构建后端镜像
    log_info "📦 Building backend image..."
    log_info "   This may take 60-90 minutes..."
    log_info "   Using Dockerfile: backend/Dockerfile.offline"
    echo ""

    DOCKER_BUILDKIT=1 docker buildx build \
        --platform linux/$PLATFORM \
        --file backend/Dockerfile.offline \
        --tag tianshu-backend:latest \
        --load \
        .

    log_success "Backend image built successfully"
    echo ""

    # 5. 构建前端镜像
    log_info "📦 Building frontend image..."
    echo ""

    DOCKER_BUILDKIT=1 docker buildx build \
        --platform linux/$PLATFORM \
        --file frontend/Dockerfile \
        --tag tianshu-frontend:latest \
        --load \
        .

    log_success "Frontend image built successfully"
    echo ""

    # 6. 拉取 RustFS 镜像（明确指定 amd64 平台）
    log_info "📥 Pulling RustFS image (platform: linux/$PLATFORM)..."
    docker pull --platform "linux/$PLATFORM" rustfs/rustfs:latest
    log_success "RustFS image pulled successfully"
    echo ""

    # 7. 导出镜像
    log_info "💾 Exporting images..."
    mkdir -p "$OUTPUT_DIR"

    log_info "   Exporting backend image (this may take 10-20 minutes)..."
    docker save tianshu-backend:latest | gzip > "$OUTPUT_DIR/tianshu-backend-$PLATFORM.tar.gz" &
    PID_BACKEND=$!

    log_info "   Exporting frontend image..."
    docker save tianshu-frontend:latest | gzip > "$OUTPUT_DIR/tianshu-frontend-$PLATFORM.tar.gz" &
    PID_FRONTEND=$!

    log_info "   Exporting rustfs image..."
    docker save rustfs/rustfs:latest | gzip > "$OUTPUT_DIR/rustfs-$PLATFORM.tar.gz" &
    PID_RUSTFS=$!

    # 等待所有导出完成
    wait $PID_BACKEND
    wait $PID_FRONTEND
    wait $PID_RUSTFS

    log_success "All images exported successfully"
    echo ""

    # 8. 处理模型文件
    log_info "📦 Handling models..."

    if [ -d "$MODELS_DIR" ] && [ ! -f "$OUTPUT_DIR/models-offline.tar.gz" ]; then
        log_info "Packaging models..."
        tar czf "$OUTPUT_DIR/models-offline.tar.gz" "$MODELS_DIR/"
        log_success "Models packaged successfully"
    else
        log_warning "Models directory not found or already packaged"
    fi
    echo ""

    # 9. 复制配置文件
    log_info "📋 Copying configuration files..."

    # 复制离线 docker-compose 为标准名称（用于部署）
    cp docker-compose.offline.yml "$OUTPUT_DIR/docker-compose.yml"
    log_info "   ✓ docker-compose.yml (from docker-compose.offline.yml)"

    # 同时保留原始文件名作为备份
    cp docker-compose.offline.yml "$OUTPUT_DIR/docker-compose.offline.yml"
    log_info "   ✓ docker-compose.offline.yml (backup)"

    # 复制 .env.example
    cp .env.example "$OUTPUT_DIR/" 2>/dev/null && log_info "   ✓ .env.example" || log_warning ".env.example not found, skipping"

    # 复制部署脚本
    cp scripts/deploy-offline.sh "$OUTPUT_DIR/" 2>/dev/null && log_info "   ✓ deploy-offline.sh" || log_warning "deploy-offline.sh not found, skipping"
    chmod +x "$OUTPUT_DIR/deploy-offline.sh" 2>/dev/null || true

    # 复制 MCP 配置示例
    if [ -f "mcp_config.example.json" ]; then
        mkdir -p "$OUTPUT_DIR/mcp_config.example.json"
        cp mcp_config.example.json "$OUTPUT_DIR/mcp_config.example.json/"
        log_info "   ✓ mcp_config.example.json"
    fi

    log_success "Configuration files copied"
    echo ""

    # 10. 显示结果
    log_info "=========================================="
    log_success "✅ Build Complete!"
    log_info "=========================================="
    echo ""
    log_info "📦 Files in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR/"
    echo ""

    # 计算总大小
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    log_info "💾 Total size: $TOTAL_SIZE"
    echo ""

    # 11. 显示下一步
    log_info "📋 Next steps:"
    echo ""
    echo "  1. Verify images:"
    echo "     docker images | grep tianshu"
    echo ""
    echo "  2. Upload all files to server (首次部署):"
    echo "     bash scripts/upload-all-to-server.sh root 192.168.1.100 /opt/tianshu"
    echo ""
    echo "  3. Or upload specific files (增量更新):"
    echo "     # 只上传后端镜像"
    echo "     bash scripts/upload-spec-to-server.sh root 192.168.1.100 /opt/tianshu backend"
    echo ""
    echo "     # 只上传前端镜像"
    echo "     bash scripts/upload-spec-to-server.sh root 192.168.1.100 /opt/tianshu frontend"
    echo ""
    echo "     # 只上传配置文件"
    echo "     bash scripts/upload-spec-to-server.sh root 192.168.1.100 /opt/tianshu config"
    echo ""
    echo "  4. Or manually transfer and deploy:"
    echo "     rsync -avz --progress $OUTPUT_DIR/ user@server:/deploy/"
    echo "     ssh user@server 'cd /deploy && bash deploy-offline.sh'"
    echo ""
    log_success "Build script completed successfully!"
}

# 捕获中断信号
trap 'log_warning "Build interrupted by user"; exit 130' SIGINT SIGTERM

# 执行主函数
main
