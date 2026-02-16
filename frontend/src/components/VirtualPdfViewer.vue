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
            :title="`定位到解析结果 (ID: ${block.id})`"
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
import { ref, computed, watch, nextTick, onUnmounted } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker

const props = defineProps<{
  src: string | null
  layoutData?: any[] 
}>()

const emit = defineEmits<{
  (e: 'block-click', block: any): void
}>()

const scrollContainer = ref<HTMLElement | null>(null)
let pdfProxy: pdfjsLib.PDFDocumentProxy | null = null

const loading = ref(false)
const processing = ref(false)
const progress = ref(0)
const error = ref<string | null>(null)
const highlightTarget = ref<{ pageIndex: number; bbox: any[] } | null>(null)

const scrollTop = ref(0)
const containerHeight = ref(0)
const totalPages = ref(0)

interface PageData {
  id: number
  width: number
  height: number
  viewport: any
  rendered: boolean
  pdfWidth: number // PDF 原生宽度，用于坐标换算
}
const pages = ref<PageData[]>([])

// 🚀 核心一：后端传来的 JSON 数据不仅包含 parsing_res_list，还有 width 和 height！
// 这些全局属性决定了后端生成 BBox 的基准坐标系。
const sourcePdfWidth = computed(() => {
  // 提取 JSON 中根节点的 width 属性，如果没有则默认为 A4 (约 595 磅)
  if (props.layoutData && props.layoutData.length > 0 && props.layoutData[0]._root_width) {
    return props.layoutData[0]._root_width;
  }
  return 595.28; 
})

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
    rootMargin: '300px 0px', 
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

const buildPageSkeletons = async () => {
  if (!pdfProxy || !scrollContainer.value) return
  processing.value = true

  const containerW = scrollContainer.value.clientWidth - 40
  if (containerW <= 0) {
    setTimeout(buildPageSkeletons, 100)
    return
  }
  containerHeight.value = scrollContainer.value.clientHeight

  const newPages: PageData[] = []
  
  // 基准第一页，用于定宽
  const page1 = await pdfProxy.getPage(1)
  const baseViewport = page1.getViewport({ scale: 1 })
  const fitScale = Math.min(containerW / baseViewport.width, 1.8) 

  for (let i = 1; i <= totalPages.value; i++) {
    const p = await pdfProxy.getPage(i)
    const vp = p.getViewport({ scale: fitScale })
    // 保存原始宽度用于比率计算
    const rawVp = p.getViewport({ scale: 1 }) 
    newPages.push({ id: i, width: vp.width, height: vp.height, viewport: vp, rendered: false, pdfWidth: rawVp.width })
  }
  
  pages.value = newPages
  processing.value = false
  initObserver()
}

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

// 🚀 核心二：绝对精准的坐标系换算算法
const getBlockStyle = (pageId: number, bbox: any) => {
  if (!bbox || !Array.isArray(bbox) || bbox.length === 0) return { display: 'none' }
  
  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  
  if (bbox.length === 4 && typeof bbox[0] === 'number') {
    [x0, y0, x1, y1] = bbox as number[];
  } else if (bbox.length === 4 && Array.isArray(bbox[0])) {
    const xs = bbox.map((p: number[]) => p[0]); const ys = bbox.map((p: number[]) => p[1]);
    x0 = Math.min(...xs); y0 = Math.min(...ys); x1 = Math.max(...xs); y1 = Math.max(...ys);
  } else { return { display: 'none' } }

  const pageInfo = pages.value.find(p => p.id === pageId)
  if (!pageInfo) return { display: 'none' }

  // 后端的 BBox 坐标是以 sourcePdfWidth 为基准的。
  // 前端 Canvas 的渲染宽度是 pageInfo.width。
  // 所以：前端缩放率 = 前端宽度 / 后端基准宽度
  const scaleRatio = pageInfo.width / sourcePdfWidth.value;

  const finalX0 = x0 * scaleRatio;
  const finalY0 = y0 * scaleRatio;
  const finalW = Math.max((x1 - x0) * scaleRatio, 5); // 最小预留 5px
  const finalH = Math.max((y1 - y0) * scaleRatio, 5);
  
  return { 
    left: `${finalX0}px`, 
    top: `${finalY0}px`, 
    width: `${finalW}px`, 
    height: `${finalH}px` 
  }
}

const highlightBlock = (pageIndex: number, bbox: any) => {
  if (!scrollContainer.value) return
  highlightTarget.value = { pageIndex, bbox }
  
  const pageNode = document.getElementById(`pdf-page-${pageIndex}`)
  if (pageNode) {
    let blockY = 0
    if (bbox && bbox.length === 4) {
      blockY = typeof bbox[0] === 'number' ? bbox[1] : Math.min(...bbox.map((p:any)=>p[1]))
    }
    const pageInfo = pages.value.find(p => p.id === pageIndex)
    const scaleRatio = pageInfo ? (pageInfo.width / sourcePdfWidth.value) : 1;
    
    // 滚动时，将目标位置放在视口正中间稍偏上
    const targetScroll = pageNode.offsetTop + (blockY * scaleRatio) - (scrollContainer.value.clientHeight / 3)
    scrollContainer.value.scrollTo({ top: Math.max(0, targetScroll), behavior: 'smooth' })
    
    setTimeout(() => { highlightTarget.value = null }, 3000)
  }
}

const onBlockClick = (block: any) => { emit('block-click', block) }
const onScroll = (e: Event) => { scrollTop.value = (e.target as HTMLElement).scrollTop }

const currentPage = computed(() => {
  if (pages.value.length === 0) return 0
  const center = scrollTop.value + (containerHeight.value / 3)
  const page = pages.value.find(p => {
     const pageTop = (document.getElementById(`pdf-page-${p.id}`)?.offsetTop) || 0;
     return center >= pageTop && center <= pageTop + p.height + PAGE_GAP;
  })
  return page ? page.id : 1
})

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
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
