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
      </div>

      <div class="flex items-center gap-3">
        <div v-if="task?.status === 'completed' && task?.data?.pdf_path" class="flex items-center bg-gray-100 rounded-lg p-1">
          <button
            @click="layoutMode = 'single'"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'single' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <FileText class="w-3.5 h-3.5 mr-1.5" />
            单页
          </button>
          <button
            @click="layoutMode = 'split'"
            :class="['px-3 py-1.5 text-xs font-medium rounded-md transition-all flex items-center', layoutMode === 'split' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500 hover:text-gray-700']"
          >
            <Columns class="w-3.5 h-3.5 mr-1.5" />
            分屏对比
          </button>
        </div>

        <button @click="refreshTask()" :disabled="loading" class="btn btn-secondary btn-sm">
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
      <div v-if="task.status !== 'completed'" class="max-w-3xl mx-auto mt-10 space-y-6 px-4">
         <div class="card p-8 text-center">
            <h2 class="text-xl font-semibold text-gray-900 mb-2">任务处理中</h2>
            <p class="text-gray-500">请稍候，解析完成后将自动显示结果...</p>
            <div class="mt-6 flex justify-center">
               <LoadingSpinner />
            </div>
         </div>
      </div>

      <div v-else class="h-full flex flex-col">
        <div :class="['flex-1 min-h-0 grid gap-4 h-full', layoutMode === 'split' ? 'grid-cols-2' : 'grid-cols-1']">
          
          <div v-if="layoutMode === 'split'" class="card p-0 overflow-hidden flex flex-col h-full border-r border-gray-200">
            <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
              <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider">源文档 / 布局预览</span>
              <a v-if="pdfUrl" :href="pdfUrl" target="_blank" class="text-xs text-primary-600 hover:underline">新窗口打开</a>
            </div>
            <div class="flex-1 bg-gray-200 relative">
              <iframe 
                v-if="pdfUrl" 
                :src="pdfUrl" 
                class="absolute inset-0 w-full h-full"
                frameborder="0"
              ></iframe>
              <div v-else class="absolute inset-0 flex items-center justify-center text-gray-400">
                暂无预览文件
              </div>
            </div>
          </div>

          <div class="card p-0 overflow-hidden flex flex-col h-full">
            <div class="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
              <div class="flex items-center gap-2">
                 <span class="text-xs font-semibold text-gray-500 uppercase tracking-wider mr-2">解析结果</span>
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
              <button @click="downloadMarkdown" class="text-xs text-primary-600 hover:underline flex items-center">
                <Download class="w-3 h-3 mr-1"/> 下载
              </button>
            </div>
            
            <div class="flex-1 overflow-auto bg-white relative custom-scrollbar">
               <div v-show="activeTab === 'markdown'" class="p-6">
                  <MarkdownViewer v-if="task.data?.content" :content="task.data.content" />
                  <div v-else class="text-center py-10 text-gray-400">无 Markdown 内容</div>
               </div>
               <div v-show="activeTab === 'json'" class="p-0 h-full">
                  <JsonViewer 
                    v-if="task.data?.json_content" 
                    :data="task.data.json_content" 
                    class="h-full overflow-auto p-4"
                  />
                  <div v-else class="text-center py-10 text-gray-400">无 JSON 数据</div>
               </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import {
  ArrowLeft, AlertCircle, RefreshCw, FileText, 
  Columns, Download, Loader
} from 'lucide-vue-next'
import StatusBadge from '@/components/StatusBadge.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import MarkdownViewer from '@/components/MarkdownViewer.vue'
import JsonViewer from '@/components/JsonViewer.vue'

const { t: $t } = useI18n()
const route = useRoute()
const taskStore = useTaskStore()

const taskId = computed(() => route.params.id as string)
const task = computed(() => taskStore.currentTask)
const loading = ref(false)
const error = ref('')

const activeTab = ref<'markdown' | 'json'>('markdown')
const layoutMode = ref<'single' | 'split'>('split') // 默认开启分屏

// 计算 PDF 的 URL
const pdfUrl = computed(() => {
  if (!task.value?.data?.pdf_path) return null
  // 必须对路径进行编码，处理中文文件名
  const encodedPath = encodeURIComponent(task.value.data.pdf_path)
  return `/api/v1/files/output/${encodedPath}`
})

let stopPolling: (() => void) | null = null

async function refreshTask(format: 'markdown' | 'json' | 'both' = 'markdown') {
  loading.value = true
  error.value = ''
  try {
    await taskStore.fetchTaskStatus(taskId.value, false, 'both') // 强制获取 both 以便支持 tab 切换
  } catch (err: any) {
    error.value = err.message || $t('task.loadFailed')
  } finally {
    loading.value = false
  }
}

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

onMounted(async () => {
  await refreshTask('both')
  
  // 自动轮询
  if (task.value && ['pending', 'processing'].includes(task.value.status)) {
    stopPolling = taskStore.pollTaskStatus(taskId.value, 2000, async (updatedTask) => {
      if (['completed', 'failed'].includes(updatedTask.status)) {
        if (stopPolling) stopPolling()
      }
    })
  }
})

onUnmounted(() => {
  if (stopPolling) stopPolling()
})
</script>

<style scoped>
/* 自定义滚动条样式，防止双滚动条太丑 */
.custom-scrollbar::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: #f1f1f1;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
