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
              class="btn btn-white text-blue-600 border-gray-200 hover:bg-blue-50 btn-sm flex items-center shadow-sm transition-all"
              :title="$t('task.retryTask')"
            >
              <RotateCw class="w-4 h-4 sm:mr-1.5" />
              <span class="hidden sm:inline">{{ $t('task.retryTask') }}</span>
            </button>

            <button
              v-if="['completed', 'failed'].includes(task.status) && task.result_path !== 'CLEARED'"
              @click="initiateAction('clearCache')"
              class="btn btn-white text-orange-600 border-gray-200 hover:bg-orange-50 btn-sm flex items-center shadow-sm transition-all"
              :title="$t('task.clearCache')"
            >
              <Eraser class="w-4 h-4 sm:mr-1.5" />
              <span class="hidden sm:inline">{{ $t('task.clearCache') }}</span>
            </button>
        </template>

        <div v-if="task?.status === 'completed' && pdfUrl && task?.result_path !== 'CLEARED'" class="flex items-center bg-gray-100 rounded-lg p-1">
          <button
            @click="layoutMode = 'single'"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'single' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <FileText class="w-3.5 h-3.5 mr-1.5" />
            {{ $t('task.singlePage') }}
          </button>
          <button
            @click="layoutMode = 'split'"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'split' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <Columns class="w-3.5 h-3.5 mr-1.5" />
            {{ $t('task.splitView') }}
          </button>
        </div>

        <button @click="refreshTask()" :disabled="loading" class="btn btn-secondary btn-sm shadow-sm">
          <RefreshCw :class="{ 'animate-spin': loading }" class="w-4 h-4" />
        </button>
      </div>
    </div>

    <div v-if="loading && !task" class="flex-1 flex items-center justify-center">
      <LoadingSpinner size="lg" :text="$t('common.loading')" />
    </div>
    
    <div v-else-if="error" class="card bg-red-50 border-red-200 mx-1">
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
               <button v-if="task.status === 'failed'" @click="initiateAction('retry')" class="btn btn-primary btn-sm inline-flex items-center">
                 <RotateCw class="w-4 h-4 mr-2"/> {{ $t('task.retryTask') }}
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

      <div v-else class="h-full flex flex-col">
        <div :class="['flex-1 min-h-0 grid gap-4 h-full', layoutMode === 'split' ? 'grid-cols-2' : 'grid-cols-1']">
          
          <div v-if="layoutMode === 'split' || layoutMode === 'single'" class="card p-0 overflow-hidden flex flex-col h-full border-r border-gray-200">
            <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
              <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">{{ $t('task.sourceDocPreview') }}</span>
              <a v-if="pdfUrl" :href="pdfUrl" target="_blank" class="text-xs text-primary-600 hover:underline flex items-center">
                {{ $t('common.openInNewWindow') }} <ExternalLink class="w-3 h-3 ml-1"/>
              </a>
            </div>
            <div class="flex-1 bg-gray-100 relative overflow-hidden">
              <VirtualPdfViewer v-if="pdfUrl" :src="pdfUrl" />
              <div v-else class="absolute inset-0 flex items-center justify-center text-gray-400">
                {{ $t('task.noPreview') }}
              </div>
            </div>
          </div>

          <div v-if="layoutMode === 'split' || layoutMode !== 'single'" class="card p-0 overflow-hidden flex flex-col h-full">
            <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
              <div class="flex items-center gap-2">
                  <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider mr-2">{{ $t('task.parseResult') }}</span>
                  <div class="flex items-center bg-gray-200 rounded p-0.5">
                    <button 
                      @click="switchTab('markdown')" 
                      :class="['text-xs px-2 py-0.5 rounded transition-all', activeTab==='markdown' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500']"
                    >Markdown</button>
                    <button 
                      @click="switchTab('json')" 
                      :class="['text-xs px-2 py-0.5 rounded transition-all', activeTab==='json' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500']"
                    >JSON</button>
                  </div>
              </div>
              <button @click="downloadMarkdown" class="text-xs text-primary-600 hover:underline flex items-center" :title="$t('common.download')">
                <Download class="w-3 h-3 mr-1"/> {{ $t('common.download') }}
              </button>
            </div>
            
            <div class="flex-1 overflow-auto bg-white relative custom-scrollbar">
               <div v-show="activeTab === 'markdown'" class="p-6">
                  <MarkdownViewer v-if="task.data?.content" :content="task.data.content" />
                  <div v-else class="text-center py-10 text-gray-400">{{ $t('task.noMarkdownContent') }}</div>
               </div>
               <div v-show="activeTab === 'json'" class="p-0 h-full">
                  <JsonViewer 
                    v-if="task.data?.json_content" 
                    :data="task.data.json_content" 
                    class="h-full overflow-auto p-4"
                  />
                  <div v-else class="text-center py-10 text-gray-400">{{ $t('task.noJsonData') }}</div>
               </div>
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
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import {
  ArrowLeft, AlertCircle, RefreshCw, FileText, 
  Columns, Download, RotateCw, Eraser, Pause, Eye, ExternalLink
} from 'lucide-vue-next'
import StatusBadge from '@/components/StatusBadge.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import MarkdownViewer from '@/components/MarkdownViewer.vue'
import JsonViewer from '@/components/JsonViewer.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
// 引入新组件
import VirtualPdfViewer from '@/components/VirtualPdfViewer.vue'

const { t } = useI18n()
const route = useRoute()
const taskStore = useTaskStore()

const taskId = computed(() => route.params.id as string)
const task = computed(() => taskStore.currentTask)
const loading = ref(false)
const error = ref('')

const activeTab = ref<'markdown' | 'json'>('markdown')
const layoutMode = ref<'single' | 'split'>('split')

// 计算 PDF 的 URL (处理编码与路径)
const pdfUrl = computed(() => {
  if (task.value?.data?.pdf_path) {
    return `/api/v1/files/output/${task.value.data.pdf_path}`
  }
  if (task.value?.source_url) {
    return task.value.source_url
  }
  return null
})

let stopPolling: (() => void) | null = null

// ----------------------------------------------------------------
// 核心操作逻辑
// ----------------------------------------------------------------
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
  
  loading.value = true
  try {
    if (currentAction.value === 'retry') {
      await taskStore.retryTask(taskId.value)
      // 重试后立即刷新状态，通常会变回 pending
      await refreshTask()
      // 重启轮询
      startPolling()
    } else if (currentAction.value === 'clearCache') {
      await taskStore.clearTaskCache(taskId.value)
      // 刷新以获取 CLEARED 状态
      await refreshTask()
    }
  } catch (err: any) {
    error.value = err.message || 'Action failed'
  } finally {
    loading.value = false
    currentAction.value = null
  }
}

// ----------------------------------------------------------------
// 数据获取与轮询
// ----------------------------------------------------------------
async function refreshTask() {
  loading.value = true
  error.value = ''
  try {
    await taskStore.fetchTaskStatus(taskId.value, false, 'both')
  } catch (err: any) {
    error.value = err.message || t('task.loadFailed')
  } finally {
    loading.value = false
  }
}

function startPolling() {
  if (stopPolling) stopPolling()
  // 5秒轮询一次，降低压力
  stopPolling = taskStore.pollTaskStatus(taskId.value, 5000, (updatedTask) => {
    // 只有在完成、失败或取消时才停止轮询
    if (['completed', 'failed', 'cancelled'].includes(updatedTask.status)) {
      if (stopPolling) stopPolling()
    }
  })
}

// ----------------------------------------------------------------
// UI 辅助
// ----------------------------------------------------------------
function switchTab(tab: 'markdown' | 'json') {
  activeTab.value = tab
}

function downloadMarkdown() {
  if (!task.value?.data?.content) return
  const blob = new Blob([task.value.data.content], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = task.value.data.markdown_file || `${taskId.value}.md`
  a.click()
  URL.revokeObjectURL(url)
}

// ----------------------------------------------------------------
// 生命周期
// ----------------------------------------------------------------
onMounted(async () => {
  await refreshTask()
  if (task.value && ['pending', 'processing'].includes(task.value.status)) {
    startPolling()
  }
})

onUnmounted(() => {
  if (stopPolling) stopPolling()
})
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 8px; height: 8px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f9fafb; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
</style>
