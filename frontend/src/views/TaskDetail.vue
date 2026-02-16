<template>
  <div class="h-[calc(100vh-4rem)] flex flex-col">
    <div class="flex items-center justify-between mb-4 px-1 flex-shrink-0">
      <div class="flex items-center gap-4">
        <button
          @click="$router.back()"
          class="text-sm text-gray-600 hover:text-gray-900 flex items-center transition-colors"
        >
          <ArrowLeft class="w-4 h-4 mr-1" />
          {{ $t('common.back') }}
        </button>
        <div class="h-4 w-px bg-gray-300"></div>
        <h1 class="text-xl font-bold text-gray-900 truncate max-w-md" :title="task?.file_name">
          {{ task?.file_name || $t('task.taskDetail') }}
        </h1>
        <StatusBadge v-if="task" :status="task.status" />
        
        <span v-if="task?.result_path === 'CLEARED'" class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-500 border border-gray-200">
           <Eraser class="w-3 h-3 mr-1.5" />
           {{ $t('status.cleared') }}
        </span>
      </div>

      <div class="flex items-center gap-3">
        <template v-if="task">
            <button 
              v-if="task.status === 'failed'"
              @click="initiateAction('retry')"
              :disabled="actionLoading"
              class="btn btn-white text-blue-600 border-gray-200 hover:bg-blue-50 btn-sm flex items-center shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              :title="$t('task.retryTask')"
            >
              <RotateCw :class="{'animate-spin': actionLoading && currentAction === 'retry'}" class="w-4 h-4 sm:mr-1.5" />
              <span class="hidden sm:inline">{{ $t('task.retryTask') }}</span>
            </button>

            <button
              v-if="['completed', 'failed'].includes(task.status) && task.result_path !== 'CLEARED'"
              @click="initiateAction('clearCache')"
              :disabled="actionLoading"
              class="btn btn-white text-orange-600 border-gray-200 hover:bg-orange-50 btn-sm flex items-center shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              :title="$t('task.clearCache')"
            >
              <Eraser :class="{'animate-pulse': actionLoading && currentAction === 'clearCache'}" class="w-4 h-4 sm:mr-1.5" />
              <span class="hidden sm:inline">{{ $t('task.clearCache') }}</span>
            </button>
        </template>

        <div v-if="layoutMode === 'split'" class="flex items-center gap-2 mr-2 bg-white px-3 py-1.5 rounded-lg border border-gray-200 shadow-sm transition-all hover:border-gray-300">
          <label class="flex items-center cursor-pointer text-xs font-medium text-gray-700 select-none" :title="$t('task.syncScrollDesc') || '开启后，左右两侧将同步滚动'">
            <input type="checkbox" v-model="syncScroll" class="mr-2 rounded text-primary-600 focus:ring-primary-500 border-gray-300 transition-colors cursor-pointer">
            <span>{{ $t('task.syncScroll') || '同步滚动' }}</span>
          </label>
        </div>

        <div v-if="task?.status === 'completed' && pdfUrl && task?.result_path !== 'CLEARED'" class="flex items-center bg-gray-100 rounded-lg p-1">
          <button
            @click="setMode('single')"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'single' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <FileText class="w-3.5 h-3.5 mr-1.5" />
            {{ $t('task.singlePage') }}
          </button>
          <button
            @click="setMode('split')"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'split' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <Columns class="w-3.5 h-3.5 mr-1.5" />
            {{ $t('task.splitView') }}
          </button>
        </div>

        <button @click="refreshTask()" :disabled="loading" class="btn btn-secondary btn-sm shadow-sm" :title="$t('common.refresh')">
          <RefreshCw :class="{ 'animate-spin': loading }" class="w-4 h-4" />
        </button>
      </div>
    </div>

    <div v-if="loading && !task" class="flex-1 flex items-center justify-center">
      <LoadingSpinner size="lg" :text="$t('common.loading')" />
    </div>
    
    <div v-else-if="error" class="card bg-red-50 border-red-200 mx-1 p-4 mb-4">
      <div class="flex items-center text-red-800">
        <AlertCircle class="w-6 h-6 mr-3" /> {{ error }}
      </div>
    </div>

    <div v-else-if="task" class="flex-1 min-h-0 relative">
      
      <div v-if="['pending', 'processing', 'paused'].includes(task.status)" class="max-w-3xl mx-auto mt-16 space-y-6 px-4">
         <div class="card p-10 text-center border-gray-100 shadow-sm">
            <h2 class="text-xl font-semibold text-gray-900 mb-2">
                {{ task.status === 'paused' ? $t('status.paused') : $t('task.taskProcessing') }}
            </h2>
            <p class="text-gray-500">
                {{ task.status === 'paused' ? $t('task.taskPausedDesc') : $t('task.processingWait') }}
            </p>
            <div class="mt-8 flex justify-center">
                <div v-if="task.status === 'paused'" class="p-4 bg-amber-50 rounded-full text-amber-500 ring-8 ring-amber-50/50">
                    <Pause class="w-8 h-8" />
                </div>
                <LoadingSpinner v-else size="lg" />
            </div>
         </div>
      </div>

      <div v-else-if="['failed', 'cancelled'].includes(task.status)" class="max-w-3xl mx-auto mt-10 space-y-6 px-4">
         <div class="card p-8 text-center border-red-100 bg-red-50/50">
            <div class="flex justify-center mb-4">
               <div class="p-3 bg-red-100 rounded-full text-red-500">
                 <AlertCircle class="w-8 h-8" />
               </div>
            </div>
            <h2 class="text-xl font-semibold text-red-700 mb-2">
                {{ task.status === 'failed' ? $t('status.failed') : $t('status.cancelled') }}
            </h2>
            <div class="text-red-600 bg-white p-4 rounded-lg border border-red-200 font-mono text-sm text-left overflow-auto max-h-64 break-all shadow-sm">
                {{ task.error_message || 'Unknown error occurred' }}
            </div>
            
            <div class="mt-6 flex justify-center gap-3">
               <a v-if="pdfUrl" :href="pdfUrl" target="_blank" class="btn btn-white text-gray-600 border-gray-300 hover:bg-gray-50 btn-sm inline-flex items-center">
                 <Eye class="w-4 h-4 mr-2"/> {{ $t('task.sourceDocPreview') }}
               </a>
               <button 
                 v-if="task.status === 'failed'" 
                 @click="initiateAction('retry')" 
                 :disabled="actionLoading"
                 class="btn btn-primary btn-sm inline-flex items-center disabled:opacity-50"
               >
                 <RotateCw :class="{'animate-spin': actionLoading && currentAction === 'retry'}" class="w-4 h-4 mr-2"/> 
                 {{ $t('task.retryTask') }}
               </button>
            </div>
         </div>
      </div>

      <div v-else-if="task.result_path === 'CLEARED'" class="max-w-3xl mx-auto mt-16 px-4">
        <div class="card p-12 text-center border-gray-200 bg-gray-50/30 shadow-sm">
           <div class="flex justify-center mb-6">
             <div class="p-4 bg-gray-100 rounded-full shadow-inner">
               <Eraser class="w-12 h-12 text-gray-400" />
             </div>
           </div>
           <h2 class="text-xl font-semibold text-gray-900 mb-2">{{ $t('task.filesCleared') }}</h2>
           <p class="text-gray-500 max-w-md mx-auto">{{ $t('task.filesClearedDesc') }}</p>
           <div class="mt-8 pt-6 border-t border-gray-100 text-sm text-gray-400 flex flex-col gap-1">
             <span>Task ID: <span class="font-mono text-gray-500">{{ task.task_id }}</span></span>
             <span>{{ $t('common.duration') }}: {{ $t('task.timeInfo') }}</span>
           </div>
        </div>
      </div>

      <div v-else class="h-full w-full flex flex-row gap-4">
        
        <div v-if="showPdf" 
             :class="['card p-0 flex flex-col h-full border border-gray-200 relative shadow-sm min-w-0 transition-all duration-300', 
                      layoutMode === 'split' ? 'flex-1 basis-1/2' : 'flex-1 basis-full']">
          <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center shrink-0">
            <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">{{ $t('task.sourceDocPreview') }}</span>
            <a v-if="pdfUrl" :href="pdfUrl" target="_blank" class="text-xs text-primary-600 hover:underline flex items-center">
              {{ $t('common.openInNewWindow') }} <ExternalLink class="w-3 h-3 ml-1"/>
            </a>
          </div>
          
          <div class="flex-1 bg-gray-200 relative overflow-hidden min-h-0">
            <VirtualPdfViewer
              ref="pdfViewerRef"
              :src="pdfUrl"
              :layout-data="layoutData"
              @scroll="handlePdfScroll"
              @block-click="handleBlockClick"
            />
            <div v-if="!pdfUrl" class="absolute inset-0 flex items-center justify-center text-gray-400">
              {{ $t('task.noPreview') }}
            </div>
          </div>
        </div>

        <div v-if="showMarkdown" 
             :class="['card p-0 flex flex-col h-full shadow-sm border border-gray-200 min-w-0 transition-all duration-300', 
                      layoutMode === 'split' ? 'flex-1 basis-1/2' : 'flex-1 basis-full']">
          <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center shrink-0">
            <div class="flex items-center bg-gray-200 rounded p-0.5">
              <button @click="activeTab = 'markdown'" :class="['tab-btn', activeTab==='markdown' ? 'active' : '']">Markdown</button>
              <button @click="activeTab = 'json'" :class="['tab-btn', activeTab==='json' ? 'active' : '']">JSON</button>
            </div>
            <button @click="downloadMarkdown" class="text-xs text-primary-600 hover:underline flex items-center">
              <Download class="w-3 h-3 mr-1"/> {{ $t('common.download') }}
            </button>
          </div>
          
          <div 
            ref="markdownContainerRef"
            class="flex-1 min-h-0 overflow-y-auto overflow-x-hidden bg-white relative custom-scrollbar p-6 scroll-smooth"
            @scroll="handleMarkdownScroll"
          >
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
                  <div v-if="block.type === 'image'" class="text-gray-400 text-xs italic mb-1 flex items-center gap-1 select-none">
                    <Image class="w-3 h-3"/> [Image Content]
                  </div>
                  <div v-else-if="block.type === 'table'" class="text-gray-400 text-xs italic mb-1 flex items-center gap-1 select-none">
                    <Table class="w-3 h-3"/> [Table Content]
                  </div>
                  
                  <div class="whitespace-pre-wrap leading-relaxed max-w-full overflow-hidden">{{ block.text }}</div>
                </div>
              </div>
              
              <MarkdownViewer v-else :content="task.data?.content || ''" />
            </div>
            
            <div v-else class="h-full w-full">
              <JsonViewer :data="task.data?.json_content || {}" />
            </div>
          </div>
        </div>

      </div>
    </div>

    <ConfirmDialog
      v-model="showConfirm"
      :title="confirmTitle"
      :message="confirmMessage"
      :type="confirmType"
      @confirm="executeAction"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import {
  ArrowLeft, AlertCircle, RefreshCw, FileText, 
  Columns, Download, RotateCw, Eraser, Pause, Eye, ExternalLink, Image, Table
} from 'lucide-vue-next'
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
const actionLoading = ref(false)
const error = ref('')

const activeTab = ref<'markdown' | 'json'>('markdown')
const layoutMode = ref<'single' | 'split'>('split')
const syncScroll = ref(true) // 默认开启同步滚动
const activeBlockId = ref<number | null>(null) 

// Refs
const pdfViewerRef = ref<InstanceType<typeof VirtualPdfViewer> | null>(null)
const markdownContainerRef = ref<HTMLElement | null>(null)

// Computed Properties
const pdfUrl = computed(() => task.value?.data?.pdf_path ? `/api/v1/files/output/${task.value.data.pdf_path}` : null)
const showPdf = computed(() => layoutMode.value === 'split' || (layoutMode.value === 'single' && pdfUrl.value))
const showMarkdown = computed(() => layoutMode.value === 'split' || layoutMode.value !== 'single')

// Data Mapping: 将嵌套的 JSON 展平为 Block 列表，便于渲染和索引
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
// 强制触发 PDF 重绘机制（解决左侧空白需要滑动才显示的问题）
// =======================================================
const triggerPdfResize = () => {
  nextTick(() => {
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'))
    }, 300) // 延迟确保 DOM 已经完全展开
  })
}

// 监听布局变化或任务加载完毕，触发重绘
watch([layoutMode, showPdf], () => {
  if (showPdf.value) triggerPdfResize()
})

watch(() => task.value?.status, (newStatus, oldStatus) => {
  if (newStatus === 'completed' && oldStatus !== 'completed') {
    triggerPdfResize()
  }
})

// =======================================================
// 同步滚动逻辑 (Bi-directional Sync)
// =======================================================
let isSyncingLeft = false
let isSyncingRight = false
let syncTimeout: any = null

const clearSyncLock = () => {
    clearTimeout(syncTimeout)
    syncTimeout = setTimeout(() => {
        isSyncingLeft = false
        isSyncingRight = false
    }, 150) // 增加延迟防止动画冲突
}

// 1. PDF 滚动 -> 带动 Markdown
const handlePdfScroll = ({ scrollTop, scrollHeight, clientHeight }: any) => {
  if (!syncScroll.value || isSyncingRight || !markdownContainerRef.value) return
  
  isSyncingLeft = true
  const maxScroll = scrollHeight - clientHeight
  if (maxScroll <= 0) return
  
  const ratio = scrollTop / maxScroll
  const md = markdownContainerRef.value
  
  md.scrollTop = ratio * (md.scrollHeight - md.clientHeight)
  
  clearSyncLock()
}

// 2. Markdown 滚动 -> 带动 PDF
const handleMarkdownScroll = (e: Event) => {
  if (!syncScroll.value || isSyncingLeft || !pdfViewerRef.value) return
  
  const target = e.target as HTMLElement
  const maxScroll = target.scrollHeight - target.clientHeight
  if (maxScroll <= 0) return
  
  isSyncingRight = true
  const ratio = target.scrollTop / maxScroll
  
  if (typeof pdfViewerRef.value.scrollToPercentage === 'function') {
    pdfViewerRef.value.scrollToPercentage(ratio)
  }
  
  clearSyncLock()
}

// =======================================================
// 双向精准定位 (Bi-directional Positioning)
// =======================================================

// 1. 点击 PDF 某块 -> 右侧 Markdown 跳转并高亮
const handleBlockClick = (block: any) => {
  if (!block || !markdownContainerRef.value) return
  
  activeBlockId.value = block.id
  
  const elementId = `block-${block.id}`
  const el = document.getElementById(elementId)
  
  if (el) {
    const oldSync = syncScroll.value
    syncScroll.value = false // 临时切断同步，防止死锁
    
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    
    setTimeout(() => { syncScroll.value = oldSync }, 800)
  }
}

// 2. 点击 Markdown 某块 -> 左侧 PDF 跳转并画框
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
// Actions & Data Loading
// =======================================================

const setMode = (mode: 'single' | 'split') => layoutMode.value = mode

let stopPolling: (() => void) | null = null

async function refreshTask() {
  loading.value = true
  error.value = ''
  try {
    await taskStore.fetchTaskStatus(taskId.value, false, 'both')
    triggerPdfResize() // 数据刷新后保证PDF容器计算正确
  } catch (err: any) {
    error.value = err.message || t('task.loadFailed')
  } finally {
    loading.value = false
  }
}

function startPolling() {
  if (stopPolling) stopPolling()
  stopPolling = taskStore.pollTaskStatus(taskId.value, 3000, (updatedTask) => {
    if (['completed', 'failed', 'cancelled'].includes(updatedTask.status)) {
      if (stopPolling) stopPolling()
      if (updatedTask.status === 'completed') triggerPdfResize()
    }
  })
}

const downloadMarkdown = () => {
  if (!task.value?.data?.content) return
  const blob = new Blob([task.value.data.content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = task.value.data.markdown_file || `${taskId.value}.md`
  a.click()
  URL.revokeObjectURL(url)
}

// --- Action Dialogs ---
const showConfirm = ref(false)
const confirmTitle = ref('')
const confirmMessage = ref('')
const confirmType = ref<'info' | 'warning' | 'danger'>('info')
const currentAction = ref<'retry' | 'clearCache' | null>(null)

function initiateAction(action: 'retry' | 'clearCache') {
  currentAction.value = action
  if (action === 'retry') {
    confirmTitle.value = t('task.retryTask')
    confirmMessage.value = t('task.confirmRetry')
    confirmType.value = 'info'
  } else if (action === 'clearCache') {
    confirmTitle.value = t('task.clearCache')
    confirmMessage.value = t('task.confirmClearCache')
    confirmType.value = 'danger'
  }
  showConfirm.value = true
}

async function executeAction() {
  if (!currentAction.value) return
  actionLoading.value = true
  try {
    if (currentAction.value === 'retry') {
      await taskStore.retryTask(taskId.value)
      await refreshTask()
      startPolling()
    } else if (currentAction.value === 'clearCache') {
      await taskStore.clearTaskCache(taskId.value)
      await refreshTask()
    }
  } catch (err: any) {
    error.value = err.message || 'Action failed'
  } finally {
    actionLoading.value = false
    currentAction.value = null
  }
}

// --- Lifecycle ---
onMounted(async () => {
  await refreshTask()
  if (task.value && ['pending', 'processing'].includes(task.value.status)) {
    startPolling()
  } else {
    triggerPdfResize() // 页面挂载完成即刻补偿一次重绘
  }
})

onUnmounted(() => {
  if (stopPolling) stopPolling()
})
</script>

<style scoped>
.tab-btn { @apply text-xs px-3 py-1 rounded transition-all text-gray-500 font-medium; }
.tab-btn.active { @apply bg-white text-gray-900 shadow-sm; }

.custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f9fafb; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
</style>
