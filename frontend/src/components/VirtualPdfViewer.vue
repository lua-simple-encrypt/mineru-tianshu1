<template>
  <div class="relative w-full h-full flex flex-col bg-gray-200/50">
    <div v-if="loading || processing" class="absolute top-0 left-0 w-full h-0.5 bg-gray-200 z-50">
      <div class="h-full bg-blue-600 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(37,99,235,0.5)]" :style="{ width: `${progress}%` }"></div>
    </div>

    <div v-if="error" class="absolute inset-0 flex flex-col items-center justify-center bg-white z-50 p-6 text-center">
      <div class="bg-red-50 p-4 rounded-full mb-3">
        <AlertCircle class="h-8 w-8 text-red-500" />
      </div>
      <div class="text-gray-900 font-semibold text-lg mb-1">PDF 加载失败</div>
      <div class="text-gray-500 text-xs break-all max-w-md bg-gray-50 p-2 rounded border border-gray-100">{{ error }}</div>
      <button @click="retry" class="mt-6 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition shadow-sm text-sm font-medium">重试</button>
    </div>

    <div ref="containerRef" class="flex-1 overflow-y-auto relative w-full custom-scrollbar outline-none" @scroll="onScroll" tabindex="0">
      <div :style="{ height: totalHeight + 'px' }" class="relative w-full">
        <div v-for="page in visiblePages" :key="page.index" class="absolute left-0 w-full flex justify-center transition-opacity duration-200" :style="{ top: page.top + 'px', height: page.height + 'px' }">
          <div class="bg-white shadow-sm relative transition-shadow hover:shadow-md" :style="{ width: page.width + 'px', height: page.height + 'px' }">
            
            <div v-if="!page.rendered" class="absolute inset-0 flex items-center justify-center bg-white z-10">
              <div class="flex flex-col items-center">
                <div class="w-8 h-8 border-3 border-blue-100 border-t-blue-600 rounded-full animate-spin mb-2"></div>
                <span class="text-gray-400 text-xs font-mono font-medium">Page {{ page.index }}</span>
              </div>
            </div>
            
            <canvas :id="`pdf-canvas-${page.index}`" class="block w-full h-full relative z-0"></canvas>

            <div v-if="page.rendered && layoutDataMap[page.index]" class="absolute inset-0 z-20 pointer-events-none">
              <div
                v-for="block in layoutDataMap[page.index]"
                :key="block.id"
                class="absolute cursor-pointer pointer-events-auto border border-transparent hover:border-blue-400 hover:bg-blue-500/10 transition-colors rounded-[2px]"
                :style="getBlockStyle(page.index, block.bbox)"
                @click.stop="onBlockClick(block)"
                :title="`双向定位 (ID: ${block.id})`"
              ></div>
            </div>

            <div 
              v-if="highlight && highlight.pageIndex === page.index"
              class="absolute z-30 border-[3px] border-red-500 bg-red-500/15 animate-pulse pointer-events-none box-border rounded-[4px] shadow-[0_0_15px_rgba(239,68,68,0.6)]"
              :style="getBlockStyle(page.index, highlight.bbox)"
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

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

const props = defineProps<{
  src: string | null
  layoutData?: any[]
}>()

const emit = defineEmits<{
  (e: 'scroll', payload: { scrollTop: number; scrollHeight: number; clientHeight: number }): void
  (e: 'block-click', block: any): void
  (e: 'page-loaded', total: number): void
  (e: 'layout-ready'): void 
}>()

function debounce(fn: any, delay: number) {
  let timeoutId: any;
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

const containerRef = ref<HTMLElement | null>(null)
const pdfDoc = shallowRef<pdfjsLib.PDFDocumentProxy | null>(null)
const pagesMetaData = ref<Array<any>>([])

const totalHeight = ref(0)
const scrollTop = ref(0)
const containerHeight = ref(0)
const totalPages = ref(0)
const scale = ref(1.5)
let lastWidth = 0 

const highlight = ref<{ pageIndex: number; bbox: any[] } | null>(null)
const loading = ref(false)
const processing = ref(false)
const progress = ref(0)
const error = ref<string | null>(null)

const renderTasks = new Map<number, pdfjsLib.RenderTask>()
const renderedPages = new Set<number>()
const PAGE_GAP = 16 

const currentPage = computed(() => {
  if (!pagesMetaData.value.length) return 0
  const center = scrollTop.value + (containerHeight.value / 3)
  const page = pagesMetaData.value.find(p => center >= p.top && center <= (p.top + p.height + PAGE_GAP))
  return page ? page.index : 1
})

const layoutDataMap = computed(() => {
  if (!props.layoutData) return {}
  const map: Record<number, any[]> = {}
  props.layoutData.forEach(block => {
    const pageNum = (typeof block.page_idx === 'number' ? block.page_idx : block.page_id) + 1
    if (!map[pageNum]) map[pageNum] = []
    map[pageNum].push(block)
  })
  return map
})

// 🚀 核心：计算每页独立的坐标缩放比例，完美兼容绝对坐标系
const pageOcrScales = computed(() => {
  const scales: Record<number, number> = {};
  for (const page of pagesMetaData.value) {
    const blocks = layoutDataMap.value[page.index];
    if (!blocks || blocks.length === 0) {
      scales[page.index] = scale.value;
      continue;
    }
    let maxX = 0;
    blocks.forEach(b => {
      let x1 = 0;
      if (b.bbox.length === 4 && typeof b.bbox[0] === 'number') x1 = b.bbox[2];
      else if (b.bbox.length === 4 && Array.isArray(b.bbox[0])) x1 = Math.max(...b.bbox.map((p:any)=>p[0]));
      if (x1 > maxX) maxX = x1;
    });

    if (maxX > page.viewport.width && page.viewport.width > 0) {
      scales[page.index] = page.width / (maxX / 0.95); // 预留5%余量
    } else {
      scales[page.index] = scale.value; 
    }
  }
  return scales;
});

const visiblePages = computed(() => {
  if (pagesMetaData.value.length === 0) return []
  const startY = scrollTop.value - containerHeight.value * 1.5
  const endY = scrollTop.value + containerHeight.value * 2.5 
  const result = []
  
  for (const page of pagesMetaData.value) {
    const pageBottom = page.top + page.height
    if (pageBottom < startY) continue
    if (page.top > endY) break
    result.push({ ...page, rendered: renderedPages.has(page.index) })
  }
  return result
})

// 🚀 核心修复：白屏问题杀手，严格监听可见页，获取明确 DOM 后触发绘制
watch(visiblePages, (pages) => {
  nextTick(() => {
    pages.forEach(p => {
      if (!p.rendered && !renderTasks.has(p.index)) {
        const canvas = document.getElementById(`pdf-canvas-${p.index}`) as HTMLCanvasElement;
        if (canvas) renderPage(canvas, p);
      }
    });
  });
}, { immediate: true, deep: true })

watch(() => props.src, (val) => { if (val) loadPdf(val) })

const loadPdf = async (url: string) => {
  if (!url) return
  if (pdfDoc.value) { pdfDoc.value.destroy(); pdfDoc.value = null; }
  pagesMetaData.value = []; renderedPages.clear(); renderTasks.forEach(t => t.cancel()); renderTasks.clear();
  
  loading.value = true; error.value = null; progress.value = 5;
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
    error.value = err.message || '无法加载文档'
  } finally {
    loading.value = false
  }
}

// 轮询机制解决由于父组件未完全显示导致的宽高度计算错误
const initLayout = async (retryCount = 0) => {
  if (!pdfDoc.value || !containerRef.value) return
  
  const containerW = containerRef.value.clientWidth
  if (containerW <= 0) {
    if (retryCount < 50) setTimeout(() => initLayout(retryCount + 1), 50)
    return
  }

  processing.value = true
  renderedPages.clear()
  renderTasks.forEach(t => t.cancel())
  renderTasks.clear()
  
  try {
    containerHeight.value = containerRef.value.clientHeight
    lastWidth = containerW
    
    const page1 = await pdfDoc.value.getPage(1)
    const viewport = page1.getViewport({ scale: 1 })
    
    const targetWidth = Math.min(containerW - 32, 1200) 
    const fitScale = targetWidth / viewport.width
    scale.value = fitScale
    
    const pages = []
    let currentTop = PAGE_GAP
    
    for (let i = 1; i <= pdfDoc.value.numPages; i++) {
      const scaledViewport = page1.getViewport({ scale: fitScale })
      pages.push({
        index: i,
        width: scaledViewport.width,
        height: scaledViewport.height,
        top: currentTop,
        viewport: scaledViewport
      })
      currentTop += scaledViewport.height + PAGE_GAP
    }
    
    pagesMetaData.value = pages
    totalHeight.value = currentTop + PAGE_GAP
    
    // 布局准备完成，向父组件发送建图信号
    nextTick(() => { emit('layout-ready') })
  } catch (e) {
    console.error("Layout init failed:", e)
  } finally {
    processing.value = false
  }
}

const renderPage = async (canvas: HTMLCanvasElement, pageMeta: any) => {
  try {
    renderedPages.add(pageMeta.index)
    const page = await pdfDoc.value!.getPage(pageMeta.index)
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
    renderTasks.delete(pageMeta.index)
  } catch (err: any) {
    if (err.name !== 'RenderingCancelledException') console.warn(`Render warning:`, err)
    renderedPages.delete(pageMeta.index)
  }
}

const getBlockStyle = (pageIndex: number, bbox: any) => {
  if (!bbox || !Array.isArray(bbox) || bbox.length === 0) return { display: 'none' }
  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  
  if (bbox.length === 4 && typeof bbox[0] === 'number') {
    [x0, y0, x1, y1] = bbox as number[];
  } else if (bbox.length === 4 && Array.isArray(bbox[0])) {
    const xs = bbox.map((p: number[]) => p[0]); const ys = bbox.map((p: number[]) => p[1]);
    x0 = Math.min(...xs); y0 = Math.min(...ys); x1 = Math.max(...xs); y1 = Math.max(...ys);
  } else { return { display: 'none' } }

  const s = pageOcrScales.value[pageIndex] || scale.value; 
  return { left: `${x0 * s}px`, top: `${y0 * s}px`, width: `${Math.max((x1-x0)*s, 8)}px`, height: `${Math.max((y1-y0)*s, 8)}px` }
}

const onBlockClick = (block: any) => { emit('block-click', block); }

const onScroll = (e: Event) => {
  const target = e.target as HTMLElement
  scrollTop.value = target.scrollTop
  emit('scroll', { scrollTop: target.scrollTop, scrollHeight: target.scrollHeight, clientHeight: target.clientHeight })
}

// 获取在 PDF 容器内的绝对 Y 坐标 (供父组件构造映射使用)
const getBlockScrollY = (block: any): number => {
  const pageNum = (typeof block.page_idx === 'number' ? block.page_idx : block.page_id) + 1;
  const pageMeta = pagesMetaData.value.find(p => p.index === pageNum);
  if (!pageMeta) return 0;

  let y0 = 0;
  if (block.bbox && block.bbox.length === 4) {
      if (typeof block.bbox[0] === 'number') y0 = block.bbox[1];
      else if (Array.isArray(block.bbox[0])) y0 = Math.min(...block.bbox.map((p:any)=>p[1]));
  }
  const s = pageOcrScales.value[pageNum] || scale.value;
  return pageMeta.top + (y0 * s);
}

// 暴露 API
const scrollToY = (y: number) => {
  if (containerRef.value) containerRef.value.scrollTo({ top: Math.max(0, y), behavior: 'auto' }); // 无动画避免卡顿
}

const highlightBlock = (pageIndex: number, bbox: any) => {
  if (!containerRef.value) return
  highlight.value = { pageIndex, bbox }
  const pageMeta = pagesMetaData.value.find(p => p.index === pageIndex)
  if (pageMeta) {
    let blockY = 0;
    if (bbox && bbox.length === 4) {
       blockY = Array.isArray(bbox[0]) ? Math.min(...bbox.map((p:any) => p[1])) : bbox[1];
    }
    const s = pageOcrScales.value[pageIndex] || scale.value;
    const targetScroll = pageMeta.top + (blockY * s) - (containerHeight.value / 3);
    containerRef.value.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' })
    setTimeout(() => { highlight.value = null }, 3000)
  }
}

const retry = () => { if (props.src) loadPdf(props.src) }

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  if (containerRef.value) {
    const handleResize = debounce(() => {
      if (!containerRef.value) return
      const currentWidth = containerRef.value.clientWidth
      if (currentWidth > 0 && pagesMetaData.value.length === 0 && pdfDoc.value) {
         initLayout() 
      } else if (currentWidth > 0 && Math.abs(currentWidth - lastWidth) > 1) {
         if (!processing.value && pdfDoc.value) initLayout()
      } else if (currentWidth > 0) {
         containerHeight.value = containerRef.value.clientHeight
      }
    }, 150)
    resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.value)
  }
  if (props.src) loadPdf(props.src)
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
  if (pdfDoc.value) { pdfDoc.value.destroy(); pdfDoc.value = null; }
  renderedPages.clear(); renderTasks.forEach(t => t.cancel());
})

defineExpose({ highlightBlock, getBlockScrollY, scrollToY })
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; border: 2px solid transparent; background-clip: content-box; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }
</style>
