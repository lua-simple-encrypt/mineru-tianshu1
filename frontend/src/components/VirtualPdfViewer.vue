<template>
  <div class="relative w-full h-full flex flex-col bg-gray-100">
    <div v-if="loading || processing" class="absolute top-0 left-0 w-full h-1 bg-gray-200 z-20">
      <div class="h-full bg-blue-500 transition-all duration-300" :style="{ width: `${progress}%` }"></div>
    </div>

    <div 
      ref="containerRef" 
      class="flex-1 overflow-y-auto relative w-full custom-scrollbar" 
      @scroll="handleScroll"
    >
      <div :style="{ height: totalHeight + 'px' }" class="relative w-full">
        <div
          v-for="page in visiblePages"
          :key="page.index"
          class="absolute left-0 w-full flex justify-center"
          :style="{ 
            top: page.top + 'px', 
            height: page.height + 'px' 
          }"
        >
          <div 
            class="bg-white shadow-lg relative"
            :style="{ 
              width: page.width + 'px', 
              height: page.height + 'px' 
            }"
          >
            <div v-if="!page.rendered" class="absolute inset-0 flex items-center justify-center text-gray-300">
              <span class="loading loading-spinner loading-md"></span>
            </div>
            <canvas :ref="(el) => setCanvasRef(el, page)" class="block"></canvas>
          </div>
        </div>
      </div>
    </div>

    <div v-if="error" class="absolute inset-0 flex items-center justify-center bg-white z-50">
      <div class="text-red-500 flex flex-col items-center">
        <span class="text-lg font-bold">Error loading PDF</span>
        <span class="text-sm mt-2">{{ error }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, nextTick } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'

// 设置 Worker (这是优化性能的关键，防止 UI 卡死)
// 注意：在 Vite 中通常需要这样引入 Worker，或者配置 CDN
// 如果构建报错，请检查 pdfjs-dist 版本或使用 CDN 路径
import pdfWorker from 'pdfjs-dist/build/pdf.worker?url'
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

const props = defineProps<{
  src: string | null
}>()

const containerRef = ref<HTMLElement | null>(null)
const pdfDoc = ref<pdfjsLib.PDFDocumentProxy | null>(null)
const pagesMetaData = ref<Array<{ width: number; height: number; aspectRatio: number; top: number }>>([])
const totalHeight = ref(0)
const scrollTop = ref(0)
const containerHeight = ref(0)
const containerWidth = ref(0)

// 状态管理
const loading = ref(false)
const processing = ref(false)
const progress = ref(0)
const error = ref<string | null>(null)
const scale = ref(1.0) // 基础缩放比例

// 渲染任务缓存 (用于取消未完成的渲染)
const renderTasks = new Map<number, pdfjsLib.RenderTask>()

// 视口缓冲 (上下多渲染几页，避免白屏)
const OVERSCAN = 2 

// 1. 初始化加载 PDF
const loadPdf = async (url: string) => {
  if (!url) return
  
  loading.value = true
  error.value = null
  progress.value = 10
  
  try {
    // 使用 Blob 加载，减少内存压力
    const response = await fetch(url)
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    
    const loadingTask = pdfjsLib.getDocument(objectUrl)
    
    loadingTask.onProgress = (p) => {
      progress.value = (p.loaded / p.total) * 100
    }
    
    pdfDoc.value = await loadingTask.promise
    progress.value = 100
    
    // 初始化布局
    await initLayout()
    
  } catch (err: any) {
    console.error('PDF Load Error:', err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

// 2. 计算布局 (获取所有页面尺寸)
const initLayout = async () => {
  if (!pdfDoc.value || !containerRef.value) return
  processing.value = true
  
  try {
    const numPages = pdfDoc.value.numPages
    const pages = []
    let currentTop = 0
    
    // 获取容器宽度用于计算缩放
    containerWidth.value = containerRef.value.clientWidth || 800
    // 给页面留一点边距 (padding)
    const availableWidth = Math.max(containerWidth.value - 32, 400) 

    // 优化：先获取第一页计算大概尺寸，后续可以懒加载或者异步获取
    // 这里为了滚动条准确，我们先获取第一页，假设大部分页面大小一致
    // 如果需要精确支持混合大小 PDF，需要遍历所有页面 getViewPort (会稍慢)
    const firstPage = await pdfDoc.value.getPage(1)
    const viewport = firstPage.getViewport({ scale: 1 })
    
    // 计算适应宽度的缩放比
    const fitScale = availableWidth / viewport.width
    scale.value = fitScale

    const pageHeight = viewport.height * fitScale
    
    // 构建元数据
    for (let i = 1; i <= numPages; i++) {
      // 间距 16px
      const gap = 16 
      pages.push({
        index: i,
        width: availableWidth,
        height: pageHeight,
        aspectRatio: viewport.width / viewport.height,
        top: currentTop + gap
      })
      currentTop += pageHeight + gap
    }
    
    pagesMetaData.value = pages
    totalHeight.value = currentTop + 32 // 底部留白
    updateContainerMetrics()
    
  } finally {
    processing.value = false
  }
}

// 3. 虚拟滚动计算核心逻辑
const visiblePages = computed(() => {
  if (pagesMetaData.value.length === 0) return []
  
  const startY = scrollTop.value - (containerHeight.value * 0.5) // 上方预加载半屏
  const endY = scrollTop.value + containerHeight.value + (containerHeight.value * 0.5) // 下方预加载半屏
  
  return pagesMetaData.value.filter(page => {
    const pageBottom = page.top + page.height
    return pageBottom > startY && page.top < endY
  }).map(page => ({
    ...page,
    rendered: false // 标记位，实际渲染后在 setCanvasRef 中处理
  }))
})

// 4. 滚动处理
const handleScroll = (e: Event) => {
  const target = e.target as HTMLElement
  scrollTop.value = target.scrollTop
}

const updateContainerMetrics = () => {
  if (containerRef.value) {
    containerHeight.value = containerRef.value.clientHeight
  }
}

// 5. Canvas 渲染逻辑
const setCanvasRef = async (el: HTMLCanvasElement | null, pageMeta: any) => {
  if (!el || !pdfDoc.value) return
  
  const pageNum = pageMeta.index
  
  // 如果已经在渲染或已完成，跳过
  // 这里做一个简单的防抖检查，防止快速滚动时频繁触发
  if (el.getAttribute('data-rendered') === 'true') return
  
  try {
    const page = await pdfDoc.value.getPage(pageNum)
    
    // 重新计算精确的 Viewport (因为 initLayout 只是预估)
    // 使用 devicePixelRatio 优化高清屏显示
    const dpr = window.devicePixelRatio || 1
    const viewport = page.getViewport({ scale: scale.value })
    
    el.width = Math.floor(viewport.width * dpr)
    el.height = Math.floor(viewport.height * dpr)
    
    // 样式宽高保持 CSS 像素
    // 注意：这里我们不设置 style.width/height，因为父容器已经限制了大小
    // 我们只需要让 canvas 填满父容器
    el.style.width = '100%'
    el.style.height = '100%'
    
    const ctx = el.getContext('2d')
    if (!ctx) return
    
    // 如果有旧的渲染任务，取消它
    if (renderTasks.has(pageNum)) {
      renderTasks.get(pageNum)!.cancel()
    }
    
    const renderContext = {
      canvasContext: ctx,
      viewport: viewport,
      transform: [dpr, 0, 0, dpr, 0, 0] // 缩放矩阵
    }
    
    const renderTask = page.render(renderContext)
    renderTasks.set(pageNum, renderTask)
    
    await renderTask.promise
    
    el.setAttribute('data-rendered', 'true')
    renderTasks.delete(pageNum)
    
  } catch (err: any) {
    if (err.name !== 'RenderingCancelledException') {
      console.error(`Page ${pageNum} render error:`, err)
    }
  }
}

// 监听 Resize
let resizeObserver: ResizeObserver
onMounted(() => {
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => {
      updateContainerMetrics()
      // 如果宽度变化大，可能需要重新 initLayout (此处省略防抖逻辑)
    })
    resizeObserver.observe(containerRef.value)
    updateContainerMetrics()
  }
  
  if (props.src) {
    loadPdf(props.src)
  }
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
  // 清理所有未完成的渲染任务
  renderTasks.forEach(task => task.cancel())
  // 销毁 PDF 文档
  if (pdfDoc.value) pdfDoc.value.destroy()
})

watch(() => props.src, (newVal) => {
  if (newVal) loadPdf(newVal)
})
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f1f1f1; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
</style>
