#!/bin/bash
# ============================================================================
# Tianshu (天枢) - CPU 本地开发环境构建脚本
# ============================================================================
#
# 功能: 构建 CPU-only Docker 镜像，适用于 Mac Apple Silicon 本地开发
#
# 使用方式:
#   bash scripts/build-local-cpu.sh
#
# 要求:
#   - Docker Desktop for Mac 20.10+
#   - 启用 Rosetta 2 模拟 (Settings → General → Use Rosetta for x86_64/amd64 emulation)
#   - 已下载离线模型到 ./models-offline 目录
#
# ============================================================================

set -e  # 遇到错误立即退出

echo "🔨 Building Tianshu CPU Docker images for local development..."
echo ""

# ============================================================================
# 检查前置条件
# ============================================================================
echo "📋 Checking prerequisites..."

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "   Please install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker is installed: $(docker --version)"

# 检查 Docker Desktop 是否启用 Rosetta 2
# 注意: 这个检查在 Mac 上可能不准确,仅作提示
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "ℹ️  Detected Apple Silicon (arm64) Mac"
    echo "   Make sure you have enabled Rosetta 2 emulation in Docker Desktop:"
    echo "   Settings → General → Use Rosetta for x86_64/amd64 emulation on Apple Silicon"
    echo ""
fi

# 检查 models-offline 目录是否存在
if [ ! -d "./models-offline" ]; then
    echo "⚠️  Warning: ./models-offline directory not found!"
    echo ""
    echo "   For offline deployment, you need to download models first:"
    echo "   1. Run: python backend/download_models.py --output ./models-offline"
    echo "   2. Or copy pre-downloaded models to ./models-offline"
    echo ""
    read -p "   Do you want to continue without offline models? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Build cancelled"
        exit 1
    fi
    echo "   Continuing without offline models (will download at runtime)..."
else
    echo "✅ Offline models found: ./models-offline"
fi

echo ""
echo "============================================================================"
echo "🏗️  Building CPU Docker Images"
echo "============================================================================"
echo ""

# ============================================================================
# 构建后端 CPU 镜像
# ============================================================================
echo "📦 Building backend CPU image..."
echo "   Platform: linux/amd64 (Rosetta 2 emulation on Apple Silicon)"
echo "   Dockerfile: backend/Dockerfile.cpu"
echo "   Tag: tianshu-backend-cpu:latest"
echo ""

# 使用 docker buildx 构建 amd64 镜像
# --platform linux/amd64: 指定目标平台（Mac 通过 Rosetta 2 模拟）
# --load: 构建完成后加载到本地 Docker
docker buildx build \
    --platform linux/amd64 \
    --file backend/Dockerfile.cpu \
    --tag tianshu-backend-cpu:latest \
    --load \
    .

echo ""
echo "✅ Backend CPU image built successfully!"
echo ""

# ============================================================================
# 构建前端镜像（如果不存在）
# ============================================================================
if ! docker images | grep -q "tianshu-frontend"; then
    echo "📦 Building frontend image..."
    echo "   Platform: linux/amd64"
    echo "   Dockerfile: frontend/Dockerfile"
    echo "   Tag: tianshu-frontend:latest"
    echo ""

    docker buildx build \
        --platform linux/amd64 \
        --file frontend/Dockerfile \
        --tag tianshu-frontend:latest \
        --load \
        .

    echo ""
    echo "✅ Frontend image built successfully!"
else
    echo "ℹ️  Frontend image already exists (tianshu-frontend:latest)"
    echo "   To rebuild, run: docker rmi tianshu-frontend:latest && bash scripts/build-local-cpu.sh"
fi

echo ""
echo "============================================================================"
echo "✅ Build Complete!"
echo "============================================================================"
echo ""
echo "📊 Built images:"
docker images | grep tianshu | grep -E "(backend-cpu|frontend)" || echo "   (No images found - this is unexpected)"
echo ""
echo "🚀 Next steps:"
echo "   1. Start services: bash scripts/start-local-cpu.sh"
echo "   2. Or manually: docker-compose -f docker-compose.cpu.yml --env-file .env.cpu up -d"
echo ""
echo "📝 Notes:"
echo "   - Images use linux/amd64 platform (emulated via Rosetta 2 on Apple Silicon)"
echo "   - CPU mode is 10-20x slower than GPU mode (suitable for development only)"
echo "   - To view logs: docker-compose -f docker-compose.cpu.yml logs -f"
echo ""
