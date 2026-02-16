<template>
  <div class="h-[calc(100vh-4rem)] flex flex-col">
    <div class="flex items-center justify-between mb-4 px-1 flex-shrink-0">
      <div class="flex items-center gap-4">
        <button @click="$router.back()" class="text-sm text-gray-600 hover:text-gray-900 flex items-center transition-colors">
          <ArrowLeft class="w-4 h-4 mr-1" /> {{ $t('common.back') }}
        </button>
        <div class="h-4 w-px bg-gray-300"></div>
        <h1 class="text-xl font-bold text-gray-900 truncate max-w-md" :title="task?.file_name">{{ task?.file_name || $t('task.taskDetail') }}</h1>
        <StatusBadge v-if="task" :status="task.status" />
      </div>

      <div class="flex items-center gap-3">
        <div v-if="layoutMode === 'split'" class="flex items-center gap-2 mr-2 bg-white px-3 py-1.5 rounded-lg border border-gray-200 shadow-sm transition-all">
          <label class="flex items-center cursor-pointer text-xs font-medium text-gray-700 select-none">
            <input type="checkbox" v-model="syncScroll" class="mr-2 rounded text-primary-600 focus:ring-primary-500 border-gray-300 cursor-pointer">
            <span>{{ $t('task.syncScroll') || '同步滚动' }}</span>
          </label>
        </div>

        <div v-if="task?.status === 'completed' && pdfUrl && task?.result_path !== 'CLEARED'" class="flex items-center bg-gray-100 rounded-lg p-1">
          <button @click="setMode('single')" :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'single' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700']">
            <FileText class="w-3.5 h-3.5 mr-1.5" /> {{ $t('task.singlePage') }}
          </button>
          <button @click="setMode('split')" :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'split' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700']">
            <Columns class="w-3.5 h-3.5 mr-1.5" /> {{ $t('task.splitView') }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="loading && !task" class="flex-1 flex items-center justify-center"><LoadingSpinner size="lg" :text="$t('common.loading')" /></div>
    <div v-else-if="error" class="card bg-red-50 border-red-200 mx-1 p-4 mb-4"><div class="flex items-center text-red-800"><AlertCircle class="w-6 h-6 mr-3" /> {{ error }}</div></div>

    <div v-else-if="task" class="flex-1 min-h-0 relative">
      <div v-if="['pending', 'processing', 'paused'].includes(task.status)" class="max-w-3xl mx-auto mt-16 space-y-6 px-4">
         <div class="card p-10 text-center shadow-sm">
            <h2 class="text-xl font-semibold text-gray-900 mb-2">处理中...</h2>
            <div class="mt-8 flex justify-center"><LoadingSpinner size="lg" /></div>
         </div>
      </div>

      <div v-else class="h-full w-full flex flex-row gap-4">
        
        <div v-if="showPdf" :class="['card p-0 flex flex-col h-full border border-gray-200 relative shadow-sm min-w-0 transition-all duration-300', layoutMode === 'split' ? 'flex-1 basis-1/2' : 'flex-1 basis-full']">
          <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center shrink-0">
            <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">{{ $t('task.sourceDocPreview') }}</span>
          </div>
          
          <div class="flex-1 bg-gray-200 relative overflow-hidden min-h-0">
            <VirtualPdfViewer
              ref="pdfViewerRef"
              :src="pdfUrl"
              :layout-data="layoutData"
              @layout-ready="debouncedBuildScrollMap"
              @scroll="handlePdfScroll"
              @block-click="handleBlockClick"
            />
          </div>
        </div>

        <div v-if="showMarkdown" :class="['card p-0 flex flex-col h-full shadow-sm border border-gray-200 min-w-0 transition-all duration-300', layoutMode === 'split' ? 'flex-1 basis-1/2' : 'flex-1 basis-full']">
          <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center shrink-0">
            <div class="flex items-center bg-gray-200 rounded p-0.5">
              <button @click="activeTab = 'markdown'" :class="['tab-btn', activeTab==='markdown' ? 'active' : '']">Markdown</button>
              <button @click="activeTab = 'json'" :class="['tab-btn', activeTab==='json' ? 'active' : '']">JSON</button>
            </div>
            <button @click="downloadMarkdown" class="text-xs text-primary-600 hover:underline flex items-center">
              <Download class="w-3 h-3 mr-1"/> {{ $t('common.download') }}
            </button>
          </div>
          
          <div ref="markdownContainerRef" class="flex-1 min-h-0 overflow-y-auto overflow-x-hidden bg-white relative custom-scrollbar p-6 scroll-smooth" @scroll="handleMarkdownScroll">
            <div v-if="activeTab === 'markdown'" class="w-full">
              
              <div v-if="layoutData.length > 0" class="prose prose-sm max-w-none text-gray-700 break-words">
                <div 
                  v-for="block in layoutData" 
                  :key="block.id"
                  :id="`block-${block.id}`"
                  :data-id="block.id"
                  @click="handleMarkdownClick(block)"
                  :class="['mb-4 p-3 rounded-md transition-colors cursor-pointer border break-words w-full', 
                           activeBlockId === block.id 
                             ? 'bg-yellow-50 border-yellow-300 shadow-sm ring-1 ring-yellow-100' 
                             : 'border-transparent hover:bg-gray-50 hover:border-gray-200']"
                >
                  <div class="whitespace-pre-wrap leading-relaxed max-w-full overflow-hidden">{{ block.text }}</div>
                </div>
              </div>
              <MarkdownViewer v-else :content="task.data?.content || ''" />

            </div>
            <div v-else class="h-full w-full"><JsonViewer :data="task.data?.json_content || {}" /></div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import { ArrowLeft, AlertCircle, RefreshCw, FileText, Columns, Download, RotateCw, Eraser, Pause, Eye, ExternalLink, Image, Table } from 'lucide-vue-next'
import StatusBadge from '@/components/StatusBadge.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import MarkdownViewer from '@/components/MarkdownViewer.vue'
import JsonViewer from '@/components/JsonViewer.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import VirtualPdfViewer from '@/components/VirtualPdfViewer.vue'

const { t } = useI18n()
const route = useRoute()
const taskStore = useTaskStore()

const taskId = computed(() => route.params.id as string)
const task = computed(() => taskStore.currentTask)
const loading = ref(false)
const error = ref('')

const activeTab = ref<'markdown' | 'json'>('markdown')
const layoutMode = ref<'split' | 'single'>('split')
const syncScroll = ref(true) 
const activeBlockId = ref<number | null>(null) 

const pdfViewerRef = ref<InstanceType<typeof VirtualPdfViewer> | null>(null)
const markdownContainerRef = ref<HTMLElement | null>(null)

const pdfUrl = computed(() => task.value?.data?.pdf_path ? `/api/v1/files/output/${task.value.data.pdf_path}` : null)
const showPdf = computed(() => layoutMode.value === 'split' || (layoutMode.value === 'single' && pdfUrl.value))
const showMarkdown = computed(() => layoutMode.value === 'split' || layoutMode.value !== 'single')

// 扁平化数据供给
const layoutData = computed(() => {
  const jsonContent = task.value?.data?.json_content
  if (!jsonContent) return []
  if (Array.isArray(jsonContent)) return jsonContent 
  if (jsonContent.pages && Array.isArray(jsonContent.pages)) {
      return jsonContent.pages.flatMap((p: any) => {
          return (p.blocks || []).map((b: any) => ({
              ...b,
              page_idx: b.page_idx ?? (typeof p.page_id === 'number' ? p.page_id - 1 : 0)
          }))
      })
  }
  return []
})

// =======================================================
// 🚀 [终极进化版] 插值映射同步滑动与互锁 (避免卡顿)
// =======================================================

const scrollMap = ref<{ pdfY: number, mdY: number }[]>([]);

const buildScrollMap = () => {
    if (!pdfViewerRef.value || !markdownContainerRef.value || layoutData.value.length === 0) return;
    const newMap = [];
    for (const b of layoutData.value) {
        const mdEl = document.getElementById(`block-${b.id}`);
        if (!mdEl) continue;
        
        const mdY = mdEl.offsetTop - 24; 
        const pdfY = pdfViewerRef.value.getBlockScrollY(b) - 24;
        
        if (pdfY > 0) newMap.push({ pdfY, mdY });
    }
    newMap.sort((a, b) => a.pdfY - b.pdfY); // 按PDF高度排序
    scrollMap.value = newMap;
}

let timeoutId: any;
const debouncedBuildScrollMap = () => {
  clearTimeout(timeoutId);
  timeoutId = setTimeout(() => { nextTick(() => buildScrollMap()) }, 300);
}

watch([layoutData, activeTab, layoutMode], () => debouncedBuildScrollMap(), { deep: true })

// 互斥锁控制同步，不再使用setTimeout引起动画卡顿！
let ignoreNextPdfEvent = false;
let ignoreNextMdEvent = false;

// 线性插值查表
const interpolate = (val: number, map: any[], keyFrom: 'pdfY' | 'mdY', keyTo: 'pdfY' | 'mdY') => {
    if (map.length === 0) return 0;
    if (val <= map[0][keyFrom]) return map[0][keyTo];
    if (val >= map[map.length - 1][keyFrom]) return map[map.length - 1][keyTo];

    for (let i = 0; i < map.length - 1; i++) {
        if (val >= map[i][keyFrom] && val <= map[i+1][keyFrom]) {
            const range = map[i+1][keyFrom] - map[i][keyFrom];
            if (range === 0) return map[i][keyTo];
            const ratio = (val - map[i][keyFrom]) / range;
            return map[i][keyTo] + ratio * (map[i+1][keyTo] - map[i][keyTo]);
        }
    }
    return 0;
}

// 核心同步：左拉拉右
const handlePdfScroll = ({ scrollTop }: any) => {
  if (ignoreNextPdfEvent) { ignoreNextPdfEvent = false; return; }
  if (!syncScroll.value || !markdownContainerRef.value || scrollMap.value.length === 0) return;
  
  const targetY = interpolate(scrollTop, scrollMap.value, 'pdfY', 'mdY');
  ignoreNextMdEvent = true; // 马上设置拦截，Markdown触发滚动时直接忽略
  markdownContainerRef.value.scrollTo({ top: Math.max(0, targetY), behavior: 'auto' });
}

// 核心同步：右拉拉左
const handleMarkdownScroll = (e: Event) => {
  if (ignoreNextMdEvent) { ignoreNextMdEvent = false; return; }
  if (!syncScroll.value || !pdfViewerRef.value || scrollMap.value.length === 0) return;
  
  const target = e.target as HTMLElement;
  const targetY = interpolate(target.scrollTop, scrollMap.value, 'mdY', 'pdfY');
  ignoreNextPdfEvent = true;
  pdfViewerRef.value.scrollToY(targetY);
}

// =======================================================
// 🎯 精准定位点击高亮功能
// =======================================================

const handleBlockClick = (block: any) => {
  if (!block || !markdownContainerRef.value) return
  activeBlockId.value = block.id
  
  const el = document.getElementById(`block-${block.id}`)
  if (el) {
    const oldSync = syncScroll.value
    syncScroll.value = false // 关掉同步锁以防止平滑动画触发抖动
    
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setTimeout(() => { syncScroll.value = oldSync }, 800)
  }
}

const handleMarkdownClick = (block: any) => {
  if (!block || !pdfViewerRef.value) return
  activeBlockId.value = block.id
  
  const oldSync = syncScroll.value
  syncScroll.value = false

  const pageIndex = (typeof block.page_idx === 'number' ? block.page_idx : block.page_id) + 1
  if (typeof pdfViewerRef.value.highlightBlock === 'function') {
    pdfViewerRef.value.highlightBlock(pageIndex, block.bbox)
  }
  setTimeout(() => { syncScroll.value = oldSync }, 800)
}

// =======================================================
// 生命周期管理
// =======================================================
const setMode = (mode: 'split' | 'single') => { layoutMode.value = mode }
let stopPolling: (() => void) | null = null

async function refreshTask() {
  loading.value = true; error.value = '';
  try { await taskStore.fetchTaskStatus(taskId.value, false, 'both') } 
  catch (err: any) { error.value = err.message || t('task.loadFailed') } 
  finally { loading.value = false }
}

function startPolling() {
  if (stopPolling) stopPolling()
  stopPolling = taskStore.pollTaskStatus(taskId.value, 3000, (updatedTask) => {
    if (['completed', 'failed', 'cancelled'].includes(updatedTask.status)) stopPolling()
  })
}

onMounted(async () => {
  await refreshTask()
  if (task.value && ['pending', 'processing'].includes(task.value.status)) startPolling()
})
onUnmounted(() => { if (stopPolling) stopPolling() })
</script>

<style scoped>
.tab-btn { @apply text-xs px-3 py-1 rounded transition-all text-gray-500 font-medium; }
.tab-btn.active { @apply bg-white text-gray-900 shadow-sm; }
.custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f9fafb; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
</style>
