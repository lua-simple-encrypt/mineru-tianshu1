<template>
  <div class="relative w-full h-full flex flex-col bg-gray-200/50">
    <div v-if="loading || processing" class="absolute top-0 left-0 w-full h-0.5 bg-gray-200 z-20">
      <div class="h-full bg-blue-600 transition-all duration-300" :style="{ width: `${progress}%` }"></div>
    </div>

    <div v-if="error" class="absolute inset-0 flex flex-col items-center justify-center bg-white z-50 p-4 text-center">
      <div class="text-red-500 font-bold mb-2">PDF 加载失败</div>
      <div class="text-gray-500 text-xs break-all max-w-md">{{ error }}</div>
      <button @click="retry" class="mt-4 px-3 py-1.5 bg-blue-50 text-blue-600 rounded hover:bg-blue-100 transition text-sm">
        重试
      </button>
    </div>

    <div 
      ref="containerRef" 
      class="flex-1 overflow-y-auto relative w-full custom-scrollbar outline-none scroll-smooth" 
      @scroll="onScroll"
      tabindex="0"
    >
      <div :style="{ height: totalHeight + 'px' }" class="relative w-full">
        <div
          v-for="page in visiblePages"
          :key="page.index"
          class="absolute left-0 w-full flex justify-center transition-opacity duration-200"
          :style="{ 
            top: page.top + 'px', 
            height: page.height + 'px' 
          }"
        >
          <div 
            class="bg-white shadow-sm relative transition-shadow hover:shadow-md"
            :style="{ 
              width: page.width + 'px', 
              height: page.height + 'px' 
            }"
          >
            <div v-if="!page.rendered" class="absolute inset-0 flex items-center justify-center bg-white z-10">
              <div class="w-6 h-6 border-2 border-blue-100 border-t-blue-500 rounded-full animate-spin"></div>
              <span class="ml-2 text-gray-400 text-xs font-mono">P.{{ page.index }}</span>
            </div>
            
            <canvas 
              :ref="(el) => renderPage(el as HTMLCanvasElement, page)" 
              class="block w-full h-full"
            ></canvas>

            <div v-if="page.rendered && layoutDataMap[page.index]" class="absolute inset-0 z-20 pointer-events-none">
              <div
                v-for="block in layoutDataMap[page.index]"
                :key="block.id"
                class="absolute cursor-pointer pointer-events-auto hover:bg-blue-500/10 hover:border-blue-500 border border-transparent transition-colors"
                :style="getBlockStyle(block.bbox)"
                @click.stop="emit('block-click', block)"
                :title="`跳转到解析内容 (ID: ${block.id})`"
              ></div>
            </div>

            <div 
              v-if="highlight && highlight.pageIndex === page.index"
              class="absolute z-30 border-2 border-red-600 bg-red-500/30 animate-pulse pointer-events-none box-border"
              :style="getBlockStyle(highlight.bbox)"
            ></div>

          </div>
        </div>
      </div>
    </div>
    
    <div v-if="!loading && totalPages > 0" class="absolute bottom-4 right-6 bg-black/60 text-white px-3 py-1 rounded-full text-xs backdrop-blur-sm z-30 font-mono shadow-lg pointer-events-none select-none">
      {{ currentPage }} / {{ totalPages }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, shallowRef } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker?url'

// 设置 Worker 全局配置
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

const props = defineProps<{
  src: string | null
  layoutData?: any[] // [{id, page_idx, bbox: [x0,y0,x1,y1]}, ...]
}>()

const emit = defineEmits<{
  (e: 'scroll', payload: { scrollTop: number; scrollHeight: number; clientHeight: number }): void
  (e: 'block-click', block: any): void
  (e: 'page-loaded', total: number): void
}>()

// --- State ---
const containerRef = ref<HTMLElement | null>(null)
const pdfDoc = shallowRef<pdfjsLib.PDFDocumentProxy | null>(null)
const pagesMetaData = ref<Array<{ index: number; width: number; height: number; top: number; viewport: any }>>([])
const totalHeight = ref(0)
const scrollTop = ref(0)
const containerHeight = ref(0)
const totalPages = ref(0)
const scale = ref(1.5)

// 高亮状态
const highlight = ref<{ pageIndex: number; bbox: number[] } | null>(null)

// Loading Status
const loading = ref(false)
const processing = ref(false)
const progress = ref(0)
const error = ref<string | null>(null)

// Caches
const renderTasks = new Map<number, pdfjsLib.RenderTask>()
const renderedPages = new Set<number>()

// Constants
const PAGE_GAP = 16 

// --- Computed ---

// 计算当前可视页码
const currentPage = computed(() => {
  if (!pagesMetaData.value.length) return 0
  const center = scrollTop.value + (containerHeight.value / 2)
  const page = pagesMetaData.value.find(p => center >= p.top && center <= (p.top + p.height + PAGE_GAP))
  return page ? page.index : 1
})

// 按页码索引布局数据
const layoutDataMap = computed(() => {
  if (!props.layoutData) return {}
  const map: Record<number, any[]> = {}
  props.layoutData.forEach(block => {
    // 兼容: layout.json page_idx 通常是 0-based，PDF.js page index 是 1-based
    const pageNum = (typeof block.page_idx === 'number' ? block.page_idx : block.page_id) + 1
    if (!map[pageNum]) map[pageNum] = []
    map[pageNum].push(block)
  })
  return map
})

// 虚拟滚动核心：计算可视页面
const visiblePages = computed(() => {
  if (pagesMetaData.value.length === 0) return []
  // 预加载视口上下各 1 屏的高度
  const startY = scrollTop.value - containerHeight.value 
  const endY = scrollTop.value + containerHeight.value * 2 
  
  return pagesMetaData.value.filter(page => {
    const pageBottom = page.top + page.height
    return pageBottom > startY && page.top < endY
  }).map(page => ({
    ...page,
    rendered: renderedPages.has(page.index)
  }))
})

// --- 核心修复：监听可见页面变化，回收资源 ---
// 解决回滚白屏问题：当页面移出可视区域时，从 renderedPages 中移除，
// 这样当它再次进入可视区域时，`renderPage` 才会再次执行绘制。
watch(visiblePages, (newPages, oldPages) => {
  if (!oldPages) return;
  
  const newIndices = new Set(newPages.map(p => p.index));
  
  // 找出刚刚移出视口的页面
  oldPages.forEach(p => {
    if (!newIndices.has(p.index)) {
      // 1. 标记为未渲染
      renderedPages.delete(p.index);
      
      // 2. 如果正在渲染，取消任务
      const task = renderTasks.get(p.index);
      if (task) {
        task.cancel();
        renderTasks.delete(p.index);
      }
    }
  });
})

// --- Methods ---

const loadPdf = async (url: string) => {
  if (!url) return
  if (pdfDoc.value) {
    pdfDoc.value.destroy()
    pdfDoc.value = null
  }
  pagesMetaData.value = []
  renderedPages.clear()
  renderTasks.forEach(t => t.cancel())
  renderTasks.clear()
  
  loading.value = true
  error.value = null
  progress.value = 10
  
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    progress.value = 30

    const loadingTask = pdfjsLib.getDocument(objectUrl)
    loadingTask.onProgress = (p) => { if (p.total) progress.value = 30 + (p.loaded / p.total) * 40 }
    
    pdfDoc.value = await loadingTask.promise
    totalPages.value = pdfDoc.value.numPages
    emit('page-loaded', totalPages.value)
    progress.value = 80
    
    await initLayout()
    progress.value = 100
    URL.revokeObjectURL(objectUrl)
  } catch (err: any) {
    console.error('PDF Error:', err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const initLayout = async () => {
  if (!pdfDoc.value || !containerRef.value) return
  processing.value = true
  
  try {
    containerHeight.value = containerRef.value.clientHeight
    const containerW = containerRef.value.clientWidth
    
    const page1 = await pdfDoc.value.getPage(1)
    const viewport = page1.getViewport({ scale: 1 })
    
    // 留出 32px 左右边距
    const fitScale = (containerW - 32) / viewport.width
    scale.value = fitScale
    
    const scaledViewport = page1.getViewport({ scale: fitScale })
    const pageH = scaledViewport.height
    const pageW = scaledViewport.width
    
    const pages = []
    let currentTop = PAGE_GAP
    
    for (let i = 1; i <= pdfDoc.value.numPages; i++) {
      pages.push({
        index: i,
        width: pageW,
        height: pageH,
        top: currentTop,
        viewport: scaledViewport
      })
      currentTop += pageH + PAGE_GAP
    }
    
    pagesMetaData.value = pages
    totalHeight.value = currentTop + PAGE_GAP
  } finally {
    processing.value = false
  }
}

const renderPage = async (canvas: HTMLCanvasElement | null, pageMeta: any) => {
  if (!canvas || !pdfDoc.value) return
  
  // 检查：如果已经渲染过，且 DOM 没有被销毁（理论上 v-for 可能会复用 DOM，但我们前面做了 watch 清理）
  // 最重要的是：如果 Canvas 的宽被重置了，说明需要重绘
  const isCanvasClear = canvas.width === 0 || canvas.width === 300 // default size
  if (!isCanvasClear && renderedPages.has(pageMeta.index)) return
  if (renderTasks.has(pageMeta.index)) return

  try {
    const page = await pdfDoc.value.getPage(pageMeta.index)
    
    const dpr = window.devicePixelRatio || 1
    canvas.width = pageMeta.width * dpr
    canvas.height = pageMeta.height * dpr
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const renderTask = page.render({
      canvasContext: ctx,
      viewport: pageMeta.viewport,
      transform: [dpr, 0, 0, dpr, 0, 0]
    })
    
    renderTasks.set(pageMeta.index, renderTask)
    await renderTask.promise
    
    renderedPages.add(pageMeta.index)
    renderTasks.delete(pageMeta.index)
  } catch (err: any) {
    if (err.name !== 'RenderingCancelledException') {
      console.warn('Render warning:', err)
    }
  }
}

// 坐标转换：PDF 坐标 -> DOM 样式
// 注意：MinerU 的 layout.json 坐标通常是 [x0, y0, x1, y1]
const getBlockStyle = (bbox: number[]) => {
  if (!bbox || bbox.length < 4) return {}
  const [x0, y0, x1, y1] = bbox
  const w = x1 - x0
  const h = y1 - y0
  const s = scale.value 
  
  return {
    left: `${x0 * s}px`,
    top: `${y0 * s}px`,
    width: `${w * s}px`,
    height: `${h * s}px`
  }
}

// --- Public API ---

const onScroll = (e: Event) => {
  const target = e.target as HTMLElement
  scrollTop.value = target.scrollTop
  emit('scroll', {
    scrollTop: target.scrollTop,
    scrollHeight: target.scrollHeight,
    clientHeight: target.clientHeight
  })
}

const scrollToPercentage = (percentage: number) => {
  if (!containerRef.value) return
  const targetTop = percentage * (containerRef.value.scrollHeight - containerRef.value.clientHeight)
  containerRef.value.scrollTo({ top: targetTop, behavior: 'auto' }) 
}

const highlightBlock = (pageIndex: number, bbox: number[]) => {
  if (!containerRef.value) return
  
  highlight.value = { pageIndex, bbox }
  
  const pageMeta = pagesMetaData.value.find(p => p.index === pageIndex)
  if (pageMeta) {
    const s = scale.value
    const blockY = bbox[1] * s
    // 滚动到该元素，稍微居中一点
    const targetScroll = pageMeta.top + blockY - (containerHeight.value / 3)
    
    containerRef.value.scrollTo({
      top: targetScroll,
      behavior: 'smooth'
    })
    
    // 3秒后清除高亮
    setTimeout(() => { highlight.value = null }, 3000)
  }
}

const retry = () => { if (props.src) loadPdf(props.src) }

// --- Lifecycle ---
let resizeObserver: ResizeObserver
onMounted(() => {
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => {
        if (!processing.value) initLayout()
    })
    resizeObserver.observe(containerRef.value)
  }
  if (props.src) loadPdf(props.src)
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
  if (pdfDoc.value) pdfDoc.value.destroy()
})

watch(() => props.src, (val) => val && loadPdf(val))

defineExpose({ scrollToPercentage, highlightBlock })
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
