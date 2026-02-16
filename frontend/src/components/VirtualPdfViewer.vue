<template>
  <div class="relative w-full h-full flex flex-col bg-gray-200/80 overflow-hidden">
    <div v-if="loading || processing" class="absolute top-0 left-0 w-full h-1 bg-gray-200 z-50">
      <div class="h-full bg-primary-600 transition-all duration-300 shadow-[0_0_10px_rgba(99,102,241,0.5)]" :style="{ width: `${progress}%` }"></div>
    </div>

    <div v-if="error" class="absolute inset-0 flex flex-col items-center justify-center bg-white z-50 p-6 text-center">
      <div class="bg-red-50 p-4 rounded-full mb-3 text-red-500">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
      </div>
      <div class="text-gray-900 font-semibold text-lg mb-1">PDF 加载失败</div>
      <div class="text-gray-500 text-xs break-all max-w-md bg-gray-50 p-2 rounded border border-gray-100 mb-4">{{ error }}</div>
      <button @click="retry" class="px-5 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition shadow-sm text-sm font-medium">重新加载</button>
    </div>

    <div ref="scrollContainer" class="flex-1 overflow-y-auto w-full custom-scrollbar relative p-4 space-y-4" @scroll="onScroll">
      
      <div 
        v-for="page in pages" 
        :key="page.id"
        :id="`pdf-page-${page.id}`"
        :data-page="page.id"
        class="pdf-page-wrapper mx-auto bg-white shadow-md relative"
        :style="{ width: page.width + 'px', height: page.height + 'px' }"
      >
        <div v-if="!page.rendered" class="absolute inset-0 flex items-center justify-center bg-gray-50/50 z-10">
          <div class="w-8 h-8 border-4 border-gray-200 border-t-primary-600 rounded-full animate-spin"></div>
        </div>

        <canvas :id="`canvas-${page.id}`" class="block w-full h-full relative z-0"></canvas>

        <div v-if="page.rendered && layoutMap[page.id]" class="absolute inset-0 z-20 pointer-events-none">
          <div
            v-for="block in layoutMap[page.id]"
            :key="block.id"
            class="absolute cursor-pointer pointer-events-auto border border-transparent hover:border-blue-400 hover:bg-blue-500/15 transition-all rounded-[2px]"
            :style="getBlockStyle(page.id, block.bbox)"
            @click.stop="onBlockClick(block)"
            :title="`定位到解析结果`"
          ></div>
        </div>

        <div 
          v-if="highlightTarget && highlightTarget.pageIndex === page.id"
          class="absolute z-30 border-[3px] border-red-500 bg-red-500/20 animate-pulse pointer-events-none box-border rounded-[2px] shadow-[0_0_15px_rgba(239,68,68,0.7)]"
          :style="getBlockStyle(page.id, highlightTarget.bbox)"
        ></div>

      </div>

    </div>
    
    <div v-if="!loading && totalPages > 0" class="absolute bottom-6 right-8 bg-gray-900/75 text-white px-3 py-1.5 rounded-md text-xs backdrop-blur-md z-30 font-mono shadow-lg pointer-events-none select-none border border-white/10">
      {{ currentPage }} <span class="text-gray-400 mx-1">/</span> {{ totalPages }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

const props = defineProps<{
  src: string | null
  layoutData?: any[] // 后端返回的 JSON 段落数组
}>()

const emit = defineEmits<{
  (e: 'block-click', block: any): void
}>()

// DOM 引用
const scrollContainer = ref<HTMLElement | null>(null)
let pdfProxy: pdfjsLib.PDFDocumentProxy | null = null

// 状态
const loading = ref(false)
const processing = ref(false)
const progress = ref(0)
const error = ref<string | null>(null)
const highlightTarget = ref<{ pageIndex: number; bbox: any[] } | null>(null)

const scrollTop = ref(0)
const containerHeight = ref(0)
const totalPages = ref(0)
const scale = ref(1.0)

// 页面数据存储
interface PageData {
  id: number
  width: number
  height: number
  viewport: any
  rendered: boolean
}
const pages = ref<PageData[]>([])

// 预处理坐标缩放比 (解决因图片原尺寸过大导致的错位)
const ocrScales = ref<Record<number, number>>({})

// 将后端的扁平数组按页码归类，方便页面渲染对应框
const layoutMap = computed(() => {
  const map: Record<number, any[]> = {}
  if (!props.layoutData) return map
  props.layoutData.forEach(block => {
    const pId = (typeof block.page_idx === 'number' ? block.page_idx : block.page_id) + 1
    if (!map[pId]) map[pId] = []
    map[pId].push(block)
  })
  return map
})

// 计算可视页面，释放看不见的内存
const visiblePages = computed(() => {
  if (pages.value.length === 0) return []
  const startY = scrollTop.value - containerHeight.value * 1.5
  const endY = scrollTop.value + containerHeight.value * 2.5 
  
  let currentTop = 0
  const result = []
  
  for (const page of pages.value) {
    const pageBottom = currentTop + page.height
    if (pageBottom >= startY && currentTop <= endY) {
      result.push({ ...page, top: currentTop })
    }
    currentTop += page.height + 16 // 16px 是 gap
  }
  return result
})

const currentPage = computed(() => {
  if (pages.value.length === 0) return 0
  let currentTop = 0
  const center = scrollTop.value + (containerHeight.value / 3)
  for (const page of pages.value) {
    if (center >= currentTop && center <= currentTop + page.height + 16) {
      return page.id
    }
    currentTop += page.height + 16
  }
  return 1
})

// 🚀 核心一：IntersectionObserver 取代滚动监听，彻底解决白屏
let observer: IntersectionObserver | null = null
const renderTasks = new Map<number, any>()

const initObserver = () => {
  if (observer) observer.disconnect()
  observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const pageId = Number((entry.target as HTMLElement).dataset.page)
      const page = pages.value.find(p => p.id === pageId)
      if (!page) return

      if (entry.isIntersecting) {
        if (!page.rendered && !renderTasks.has(pageId)) {
          renderCanvas(page)
        }
      }
    })
  }, {
    root: scrollContainer.value,
    rootMargin: '200px 0px', 
    threshold: 0.01
  })

  nextTick(() => {
    const pageNodes = scrollContainer.value?.querySelectorAll('.pdf-page-wrapper')
    pageNodes?.forEach(node => observer?.observe(node))
  })
}

const loadPdf = async (url: string) => {
  if (!url) return
  error.value = null; loading.value = true; progress.value = 10;
  pages.value = []; renderTasks.clear();
  if (pdfProxy) { pdfProxy.destroy(); pdfProxy = null }

  try {
    const loadingTask = pdfjsLib.getDocument(url)
    loadingTask.onProgress = (p) => { if (p.total) progress.value = 10 + (p.loaded / p.total) * 60 }
    pdfProxy = await loadingTask.promise
    totalPages.value = pdfProxy.numPages
    progress.value = 80
    await buildPageSkeletons()
  } catch (err: any) {
    error.value = 'PDF解析失败，请检查文件格式。'
  } finally {
    loading.value = false; progress.value = 100
  }
}

// 构建所有页面的高度骨架
const buildPageSkeletons = async () => {
  if (!pdfProxy || !scrollContainer.value) return
  processing.value = true

  const containerW = scrollContainer.value.clientWidth - 32
  // 如果宽度未分配，重试
  if (containerW <= 0) {
    setTimeout(buildPageSkeletons, 100)
    return
  }
  containerHeight.value = scrollContainer.value.clientHeight

  const newPages: PageData[] = []
  const page1 = await pdfProxy.getPage(1)
  const baseViewport = page1.getViewport({ scale: 1 })
  const fitScale = Math.min(containerW / baseViewport.width, 1.8)
  scale.value = fitScale

  for (let i = 1; i <= totalPages.value; i++) {
    const vp = i === 1 ? page1.getViewport({ scale: fitScale }) : (await pdfProxy.getPage(i)).getViewport({ scale: fitScale })
    newPages.push({ id: i, width: vp.width, height: vp.height, viewport: vp, rendered: false })
  }
  
  pages.value = newPages
  calculateOcrScales(newPages, fitScale)
  processing.value = false
  initObserver()
}

// 渲染 Canvas
const renderCanvas = async (page: PageData) => {
  if (!pdfProxy) return
  renderTasks.set(page.id, true)
  
  try {
    const pdfPage = await pdfProxy.getPage(page.id)
    const canvas = document.getElementById(`canvas-${page.id}`) as HTMLCanvasElement
    if (!canvas) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = page.width * dpr
    canvas.height = page.height * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const renderCtx = { canvasContext: ctx, viewport: page.viewport, transform: [dpr, 0, 0, dpr, 0, 0] }
    await pdfPage.render(renderCtx).promise
    page.rendered = true
  } catch (err: any) {
    if (err.name !== 'RenderingCancelledException') console.warn(`Render Page ${page.id} failed:`, err)
  } finally {
    renderTasks.delete(page.id)
  }
}

// 🚀 核心二：计算绝对坐标转换缩放比，保证红框不乱飞
const calculateOcrScales = (pageList: PageData[], baseScale: number) => {
  const scales: Record<number, number> = {}
  for (const p of pageList) {
    const blocks = layoutMap.value[p.id]
    if (!blocks || blocks.length === 0) {
      scales[p.id] = baseScale; continue;
    }
    
    let maxX = 0
    blocks.forEach(b => {
      let x1 = 0
      if (b.bbox.length === 4 && typeof b.bbox[0] === 'number') x1 = b.bbox[2]
      else if (b.bbox.length === 4 && Array.isArray(b.bbox[0])) x1 = Math.max(...b.bbox.map((pt:any)=>pt[0]))
      if (x1 > maxX) maxX = x1
    })

    // PDF.js 底层 baseWidth
    const basePdfWidth = p.viewport.width / baseScale;
    
    // 如果返回的坐标远大于原始尺寸，说明后端的 bbox 是基于大图生成的
    if (maxX > basePdfWidth + 10) {
      // ratio: 将大图坐标缩放回 PDF原始尺寸 的比例
      const ratio = basePdfWidth / (maxX / 0.95); 
      scales[p.id] = ratio * baseScale;
    } else {
      scales[p.id] = baseScale
    }
  }
  ocrScales.value = scales
}

const getBlockStyle = (pageId: number, bbox: any) => {
  if (!bbox || !Array.isArray(bbox) || bbox.length === 0) return { display: 'none' }
  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  
  if (bbox.length === 4 && typeof bbox[0] === 'number') {
    [x0, y0, x1, y1] = bbox as number[];
  } else if (bbox.length === 4 && Array.isArray(bbox[0])) {
    const xs = bbox.map((p: number[]) => p[0]); const ys = bbox.map((p: number[]) => p[1]);
    x0 = Math.min(...xs); y0 = Math.min(...ys); x1 = Math.max(...xs); y1 = Math.max(...ys);
  } else { return { display: 'none' } }

  const s = ocrScales.value[pageId] || scale.value; 
  return { 
    left: `${x0 * s}px`, top: `${y0 * s}px`, 
    width: `${Math.max((x1-x0)*s, 4)}px`, height: `${Math.max((y1-y0)*s, 4)}px` 
  }
}

// 暴露 API
const highlightBlock = (pageIndex: number, bbox: any) => {
  if (!scrollContainer.value) return
  highlightTarget.value = { pageIndex, bbox }
  
  const pageNode = document.getElementById(`pdf-page-${pageIndex}`)
  if (pageNode) {
    let blockY = 0
    if (bbox && bbox.length === 4) {
      blockY = typeof bbox[0] === 'number' ? bbox[1] : Math.min(...bbox.map((p:any)=>p[1]))
    }
    const s = ocrScales.value[pageIndex] || scale.value;
    const targetScroll = pageNode.offsetTop + (blockY * s) - (scrollContainer.value.clientHeight / 4)
    scrollContainer.value.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' })
    
    setTimeout(() => { highlightTarget.value = null }, 3000)
  }
}

const onBlockClick = (block: any) => {
  emit('block-click', block)
}

const onScroll = (e: Event) => {
  scrollTop.value = (e.target as HTMLElement).scrollTop
}

const retry = () => { if (props.src) loadPdf(props.src) }

watch(() => props.src, (url) => { if(url) loadPdf(url) }, { immediate: true })

onUnmounted(() => {
  if (observer) observer.disconnect()
  if (pdfProxy) { pdfProxy.destroy(); pdfProxy = null }
  renderTasks.clear()
})

defineExpose({ highlightBlock })
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; border: 2px solid transparent; background-clip: content-box; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }
</style>
