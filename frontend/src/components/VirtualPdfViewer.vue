<template>
  <div class="relative w-full h-full flex flex-col bg-gray-200/50">
    <div v-if="loading || processing" class="absolute top-0 left-0 w-full h-0.5 bg-gray-200 z-50">
      <div class="h-full bg-blue-600 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(37,99,235,0.5)]" :style="{ width: `${progress}%` }"></div>
    </div>

    <div v-if="error" class="absolute inset-0 flex flex-col items-center justify-center bg-white z-50 p-6 text-center">
      <div class="bg-red-50 p-4 rounded-full mb-3">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      </div>
      <div class="text-gray-900 font-semibold text-lg mb-1">PDF 加载失败</div>
      <div class="text-gray-500 text-xs break-all max-w-md bg-gray-50 p-2 rounded border border-gray-100">{{ error }}</div>
      <button 
        @click="retry" 
        class="mt-6 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition shadow-sm text-sm font-medium flex items-center"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.058M20.9 14.25a8.5 8.5 0 11-6.1-1.53M22 22v-5h-5" />
        </svg>
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
              <div class="flex flex-col items-center">
                <div class="w-8 h-8 border-3 border-blue-100 border-t-blue-600 rounded-full animate-spin mb-2"></div>
                <span class="text-gray-400 text-xs font-mono font-medium">Page {{ page.index }}</span>
              </div>
            </div>
            
            <canvas 
              :ref="(el) => renderPage(el as HTMLCanvasElement, page)" 
              class="block w-full h-full"
            ></canvas>

            <div v-if="page.rendered && layoutDataMap[page.index]" class="absolute inset-0 z-20 pointer-events-none">
              <div
                v-for="block in layoutDataMap[page.index]"
                :key="block.id"
                class="absolute cursor-pointer pointer-events-auto hover:bg-blue-600/10 hover:border-blue-500 border border-transparent transition-colors rounded-[1px]"
                :style="getBlockStyle(block.bbox)"
                @click.stop="emit('block-click', block)"
                :title="`跳转到解析内容 (ID: ${block.id})`"
              ></div>
            </div>

            <div 
              v-if="highlight && highlight.pageIndex === page.index"
              class="absolute z-30 border-2 border-red-500 bg-red-500/20 animate-pulse pointer-events-none box-border rounded-[1px]"
              :style="getBlockStyle(highlight.bbox)"
            ></div>

          </div>
        </div>
      </div>
    </div>
    
    <div v-if="!loading && totalPages > 0" class="absolute bottom-6 right-8 bg-gray-900/75 text-white px-3 py-1.5 rounded-md text-xs backdrop-blur-md z-30 font-mono shadow-lg pointer-events-none select-none border border-white/10">
      {{ currentPage }} <span class="text-gray-400 mx-1">/</span> {{ totalPages }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed, shallowRef, nextTick } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker?url'

// 设置 Worker 全局配置
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

// --- Types & Props ---
const props = defineProps<{
  src: string | null
  layoutData?: any[] // [{id, page_idx, bbox: [x0,y0,x1,y1]}, ...]
}>()

const emit = defineEmits<{
  (e: 'scroll', payload: { scrollTop: number; scrollHeight: number; clientHeight: number }): void
  (e: 'block-click', block: any): void
  (e: 'page-loaded', total: number): void
}>()

// --- Utilities ---
// 简单的防抖函数
function debounce<T extends (...args: any[]) => void>(fn: T, delay: number) {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

// --- State ---
const containerRef = ref<HTMLElement | null>(null)
const pdfDoc = shallowRef<pdfjsLib.PDFDocumentProxy | null>(null)
const pagesMetaData = ref<Array<{ index: number; width: number; height: number; top: number; viewport: any }>>([])

// Layout State
const totalHeight = ref(0)
const scrollTop = ref(0)
const containerHeight = ref(0)
const totalPages = ref(0)
const scale = ref(1.5)
let lastWidth = 0 // 记录上一次的宽度，防止高度引起的无效重绘

// Highlight State
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

// 按页码索引布局数据 (优化：添加 ?. 保护)
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

// 虚拟滚动核心：计算可视页面 (优化：Early Break 循环)
const visiblePages = computed(() => {
  if (pagesMetaData.value.length === 0) return []
  
  // 预加载视口上下各 1.5 屏的高度，提升快速滚动体验
  const startY = scrollTop.value - containerHeight.value * 1.5
  const endY = scrollTop.value + containerHeight.value * 2.5 
  
  const result = []
  
  // 因为 pagesMetaData 是按 top 排序的，可以使用循环并在超出范围后 break
  for (const page of pagesMetaData.value) {
    const pageBottom = page.top + page.height
    if (pageBottom < startY) continue
    if (page.top > endY) break
    
    result.push({
      ...page,
      rendered: renderedPages.has(page.index)
    })
  }
  
  return result
})

// --- Watchers ---

// 资源回收：当页面移出视口时，取消渲染任务并移除缓存状态
watch(visiblePages, (newPages, oldPages) => {
  if (!oldPages) return;
  
  const newIndices = new Set(newPages.map(p => p.index));
  
  oldPages.forEach(p => {
    if (!newIndices.has(p.index)) {
      // 1. 标记为未渲染
      renderedPages.delete(p.index);
      
      // 2. 强行取消正在进行的渲染任务
      const task = renderTasks.get(p.index);
      if (task) {
        task.cancel();
        renderTasks.delete(p.index);
      }
    }
  });
})

// 监听源文件变化
watch(() => props.src, (val) => {
  if (val) loadPdf(val)
})

// --- Methods ---

const loadPdf = async (url: string) => {
  if (!url) return
  
  // Reset
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
  progress.value = 5
  
  try {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    
    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    progress.value = 20

    const loadingTask = pdfjsLib.getDocument(objectUrl)
    loadingTask.onProgress = (p) => { if (p.total) progress.value = 20 + (p.loaded / p.total) * 60 }
    
    pdfDoc.value = await loadingTask.promise
    totalPages.value = pdfDoc.value.numPages
    emit('page-loaded', totalPages.value)
    progress.value = 90
    
    await initLayout()
    progress.value = 100
    URL.revokeObjectURL(objectUrl)
  } catch (err: any) {
    console.error('PDF Load Error:', err)
    error.value = err.message || '无法加载文档'
  } finally {
    loading.value = false
  }
}

const initLayout = async () => {
  if (!pdfDoc.value || !containerRef.value) return
  
  const containerW = containerRef.value.clientWidth
  // 🚨【关键修复】：防止因页面尚未完全挂载（display:none 或外层未渲染完）导致 clientWidth 为 0 造成的白屏
  if (containerW <= 0) {
    setTimeout(() => {
      if (pdfDoc.value) initLayout()
    }, 100)
    return
  }

  processing.value = true
  
  // 重新计算布局时，必须清除旧的渲染状态
  renderedPages.clear()
  renderTasks.forEach(t => t.cancel())
  renderTasks.clear()
  
  try {
    containerHeight.value = containerRef.value.clientHeight
    lastWidth = containerW
    
    // 获取第一页以计算缩放比例
    const page1 = await pdfDoc.value.getPage(1)
    const viewport = page1.getViewport({ scale: 1 })
    
    // 留出 32px (左右 padding) + 滚动条空间，限制最大宽度
    const targetWidth = Math.min(containerW - 32, 1200) 
    const fitScale = targetWidth / viewport.width
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
  } catch (e) {
    console.error("Layout init failed:", e)
  } finally {
    processing.value = false
  }
}

const renderPage = async (canvas: HTMLCanvasElement | null, pageMeta: any) => {
  if (!canvas || !pdfDoc.value) return
  
  // 防御性：避免同一个 Canvas 触发多次不必要的渲染
  if (renderedPages.has(pageMeta.index) || renderTasks.has(pageMeta.index)) return

  try {
    // 立即标记占位，防止 Vue 生命周期并发调用
    renderedPages.add(pageMeta.index)
    
    const page = await pdfDoc.value.getPage(pageMeta.index)
    
    const dpr = window.devicePixelRatio || 1
    canvas.width = pageMeta.width * dpr
    canvas.height = pageMeta.height * dpr
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 渲染任务
    const renderTask = page.render({
      canvasContext: ctx,
      viewport: pageMeta.viewport,
      transform: [dpr, 0, 0, dpr, 0, 0]
    })
    
    renderTasks.set(pageMeta.index, renderTask)
    await renderTask.promise
    renderTasks.delete(pageMeta.index)
    
  } catch (err: any) {
    if (err.name !== 'RenderingCancelledException') {
      console.warn(`Page ${pageMeta.index} render warning:`, err)
    }
    // 渲染失败或被取消时，移除标记以便下次重新渲染
    renderedPages.delete(pageMeta.index)
  }
}

// 坐标转换：PDF 坐标 -> DOM 样式
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

// API: 滚动到指定百分比 (Phase 4)
const scrollToPercentage = (percentage: number) => {
  if (!containerRef.value) return
  const targetTop = percentage * (containerRef.value.scrollHeight - containerRef.value.clientHeight)
  // 使用 auto 避免同步时的延迟累积
  containerRef.value.scrollTo({ top: targetTop, behavior: 'auto' }) 
}

// API: 高亮并跳转到指定区域 (Phase 5)
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
    
    // 3秒后自动清除高亮
    setTimeout(() => { highlight.value = null }, 3000)
  }
}

const retry = () => { if (props.src) loadPdf(props.src) }

// --- Lifecycle ---
let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  if (containerRef.value) {
    // 🚨【关键修复】：只有在宽度真正发生变化时才触发重绘，防止循环重绘
    const handleResize = debounce(() => {
      if (!containerRef.value) return
      const currentWidth = containerRef.value.clientWidth
      if (currentWidth > 0 && Math.abs(currentWidth - lastWidth) > 1) {
        if (!processing.value && pdfDoc.value) {
           initLayout()
        }
      } else if (currentWidth > 0) {
        // 如果只是高度变化，仅更新高度参考值供滚动计算使用
        containerHeight.value = containerRef.value.clientHeight
      }
    }, 200)

    resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.value)
  }
  if (props.src) loadPdf(props.src)
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
  if (pdfDoc.value) {
    pdfDoc.value.destroy()
    pdfDoc.value = null
  }
  renderedPages.clear()
  renderTasks.forEach(t => t.cancel())
})

// 暴露给父组件的方法
defineExpose({ scrollToPercentage, highlightBlock })
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; border: 2px solid transparent; background-clip: content-box; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }
</style>
