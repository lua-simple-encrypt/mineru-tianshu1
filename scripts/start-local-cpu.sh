#!/bin/bash
# ============================================================================
# Tianshu (天枢) - CPU 本地开发环境启动脚本
# ============================================================================
#
# 功能: 启动 CPU-only Docker Compose 服务
#
# 使用方式:
#   bash scripts/start-local-cpu.sh
#
# 要求:
#   - 已运行 bash scripts/build-local-cpu.sh 构建镜像
#   - Docker Desktop for Mac 20.10+
#
# ============================================================================

set -e  # 遇到错误立即退出

echo "🚀 Starting Tianshu CPU services for local development..."
echo ""

# ============================================================================
# 检查前置条件
# ============================================================================
echo "📋 Checking prerequisites..."

# 检查 Docker 是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi

echo "✅ Docker is running"

# 检查 CPU 镜像是否存在
if ! docker images | grep -q "tianshu-backend-cpu"; then
    echo "❌ CPU backend image not found!"
    echo "   Please build the image first: bash scripts/build-local-cpu.sh"
    exit 1
fi

echo "✅ CPU backend image found"

# ============================================================================
# 生成或检查 .env.cpu 配置文件
# ============================================================================
if [ ! -f ".env.cpu" ]; then
    echo ""
    echo "📝 Generating .env.cpu configuration file..."

    # 复制示例配置
    if [ -f ".env.example" ]; then
        cp .env.example .env.cpu
        echo "   ✅ Copied from .env.example"
    else
        echo "   ❌ .env.example not found, creating minimal config..."
        # 创建最小配置（如果示例不存在）
        cat > .env.cpu << 'EOF'
# Tianshu CPU Development Environment
API_PORT=8000
WORKER_PORT=8001
FRONTEND_PORT=80
LOG_LEVEL=INFO

# JWT Configuration
JWT_SECRET_KEY=temp-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:80

# CPU Mode
ACCELERATOR=cpu
GPU_COUNT=0
CUDA_VISIBLE_DEVICES=
WORKER_GPUS=0
MAX_BATCH_SIZE=1
WORKER_TIMEOUT=600
WORKER_MEMORY_LIMIT=8G
WORKER_MEMORY_RESERVATION=4G

# Models
MODEL_DOWNLOAD_SOURCE=local
HF_OFFLINE=1
HF_ENDPOINT=https://hf-mirror.com

# RustFS
RUSTFS_ENABLED=true
RUSTFS_ENDPOINT=rustfs:9000
RUSTFS_ACCESS_KEY=rustfsadmin
RUSTFS_SECRET_KEY=rustfsadmin
RUSTFS_BUCKET=ts-img
RUSTFS_SECURE=false
RUSTFS_PUBLIC_URL=http://host.docker.internal:9000
RUSTFS_PORT=9000
RUSTFS_CONSOLE_PORT=9001

# Database
DATABASE_PATH=/app/data/db/mineru_tianshu.db

# Frontend
VITE_API_BASE_URL=http://localhost:8000
EOF
    fi

    # 生成随机 JWT secret
    echo "   🔐 Generating JWT secret key..."
    if command -v openssl &> /dev/null; then
        JWT_SECRET=$(openssl rand -hex 32)
        # 使用 sed 替换 JWT_SECRET_KEY
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed 需要 -i ''
            sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env.cpu
        else
            # Linux sed
            sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env.cpu
        fi
        echo "   ✅ JWT secret generated"
    else
        echo "   ⚠️  openssl not found, using default JWT secret (not secure for production)"
    fi

    # 确保 CPU 模式配置正确
    echo "   🔧 Applying CPU mode overrides..."

    # 检查并添加 CPU 配置（如果不存在）
    if ! grep -q "^ACCELERATOR=cpu" .env.cpu; then
        echo "" >> .env.cpu
        echo "# CPU Mode Overrides (Auto-generated)" >> .env.cpu
        echo "ACCELERATOR=cpu" >> .env.cpu
    fi

    if ! grep -q "^CUDA_VISIBLE_DEVICES=" .env.cpu; then
        echo "CUDA_VISIBLE_DEVICES=" >> .env.cpu
    fi

    if ! grep -q "^MODEL_DOWNLOAD_SOURCE=" .env.cpu; then
        echo "MODEL_DOWNLOAD_SOURCE=local" >> .env.cpu
    fi

    if ! grep -q "^HF_OFFLINE=" .env.cpu; then
        echo "HF_OFFLINE=1" >> .env.cpu
    fi

    echo "   ✅ CPU configuration applied"
    echo ""
    echo "   📄 Configuration saved to .env.cpu"
    echo "   ⚠️  Please review and update if needed (especially RUSTFS_PUBLIC_URL for network access)"
else
    echo "✅ .env.cpu configuration file found"
fi

echo ""
echo "============================================================================"
echo "🚀 Starting Docker Compose services..."
echo "============================================================================"
echo ""

# ============================================================================
# 启动服务
# ============================================================================
docker-compose -f docker-compose.cpu.yml --env-file .env.cpu up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# ============================================================================
# 显示服务状态
# ============================================================================
echo ""
echo "============================================================================"
echo "📊 Service Status"
echo "============================================================================"
echo ""
docker-compose -f docker-compose.cpu.yml ps

# ============================================================================
# 健康检查
# ============================================================================
echo ""
echo "🔍 Checking service health..."
echo ""

# 检查 Backend API
if curl -f -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "⚠️  Backend API is not responding yet (may need more time to start)"
fi

# 检查 Worker
if curl -f -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Worker is healthy"
else
    echo "⚠️  Worker is not responding yet (may need more time to start)"
fi

# 检查 Frontend
if curl -f -s http://localhost/ > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "⚠️  Frontend is not responding yet (may need more time to start)"
fi

# 检查 RustFS
if curl -f -s http://localhost:9000/health > /dev/null 2>&1; then
    echo "✅ RustFS is healthy"
else
    echo "⚠️  RustFS is not responding yet (may need more time to start)"
fi

echo ""
echo "============================================================================"
echo "✅ Services Started!"
echo "============================================================================"
echo ""
echo "📊 Access URLs:"
echo "   • Frontend:       http://localhost"
echo "   • API Docs:       http://localhost:8000/docs"
echo "   • API Health:     http://localhost:8000/api/v1/health"
echo "   • Worker Health:  http://localhost:8001/health"
echo "   • RustFS Console: http://localhost:9001"
echo ""
echo "🔐 Default Login:"
echo "   • Username: admin"
echo "   • Password: admin"
echo ""
echo "📝 Useful Commands:"
echo "   • View logs:      docker-compose -f docker-compose.cpu.yml logs -f"
echo "   • View backend:   docker-compose -f docker-compose.cpu.yml logs -f backend"
echo "   • View worker:    docker-compose -f docker-compose.cpu.yml logs -f worker"
echo "   • Stop services:  docker-compose -f docker-compose.cpu.yml down"
echo "   • Restart backend: docker-compose -f docker-compose.cpu.yml restart backend"
echo "   • Restart worker: docker-compose -f docker-compose.cpu.yml restart worker"
echo ""
echo "⚠️  Performance Notes:"
echo "   • CPU mode is 10-20x slower than GPU mode"
echo "   • Suitable for development and testing only"
echo "   • First requests may take longer (model loading)"
echo ""
echo "💡 Tips:"
echo "   • Code changes in ./backend will auto-reload (API server only)"
echo "   • Worker code changes require manual restart"
echo "   • Check logs if services are not responding: docker-compose -f docker-compose.cpu.yml logs -f"
echo ""

# ============================================================================
# 检查 models-offline 目录
# ============================================================================
if [ ! -d "./models-offline" ] || [ -z "$(ls -A ./models-offline 2>/dev/null)" ]; then
    echo "⚠️  Warning: Offline models not found or empty"
    echo "   Models will be downloaded at runtime (requires internet connection)"
    echo "   To use offline mode, download models first:"
    echo "   python backend/download_models.py --output ./models-offline"
    echo ""
fi
