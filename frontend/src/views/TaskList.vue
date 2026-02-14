<template>
  <div class="max-w-6xl mx-auto px-4 py-6 animate-fade-in">
    <div class="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">{{ $t('task.taskList') }}</h1>
        <p class="mt-1 text-sm text-gray-500">{{ $t('task.taskList') }}</p>
      </div>
      
      <div class="flex flex-wrap items-center gap-3">
        <label class="flex items-center cursor-pointer bg-white px-3 py-2 rounded-lg border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors" :title="$t('common.autoRefreshLabel')">
          <input type="checkbox" v-model="autoRefresh" class="sr-only">
          <div class="relative w-8 h-4 transition-colors rounded-full" :class="autoRefresh ? 'bg-green-500' : 'bg-gray-300'">
            <div class="absolute left-0.5 top-0.5 w-3 h-3 bg-white rounded-full transition-transform shadow-sm" :class="autoRefresh ? 'translate-x-4' : 'translate-x-0'"></div>
          </div>
          <span class="ml-2 text-xs font-medium text-gray-600 select-none">{{ $t('common.autoRefreshLabel') }}</span>
        </label>

        <button
          @click="refreshTasks(true)"
          :disabled="loading"
          class="btn btn-secondary btn-sm flex items-center shadow-sm"
        >
          <RefreshCw :class="{ 'animate-spin': loading }" class="w-4 h-4 mr-1.5" />
          {{ $t('common.refresh') }}
        </button>
        
        <router-link to="/tasks/submit" class="btn btn-primary btn-sm flex items-center shadow-sm">
          <Plus class="w-4 h-4 mr-1.5" />
          {{ $t('task.submitTask') }}
        </router-link>
      </div>
    </div>

    <div class="card mb-6 shadow-sm border-gray-100">
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-1">
        <div>
          <label class="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">{{ $t('task.filterByStatus') }}</label>
          <div class="relative">
            <select
              v-model="filters.status"
              @change="applyFilters"
              class="w-full pl-3 pr-8 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors text-sm appearance-none"
            >
              <option value="">{{ $t('task.allStatus') }}</option>
              <option value="pending">{{ $t('status.pending') }}</option>
              <option value="processing">{{ $t('status.processing') }}</option>
              <option value="completed">{{ $t('status.completed') }}</option>
              <option value="failed">{{ $t('status.failed') }}</option>
              <option value="cancelled">{{ $t('status.cancelled') }}</option>
            </select>
            <Filter class="absolute right-2.5 top-2.5 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
        </div>

        <div>
          <label class="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">{{ $t('task.backend') }}</label>
          <div class="relative">
            <select
              v-model="filters.backend"
              @change="applyFilters"
              class="w-full pl-3 pr-8 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors text-sm appearance-none"
            >
              <option value="">{{ $t('task.allStatus') }}</option>
              <optgroup :label="$t('task.groupMinerU')">
                <option value="pipeline">Pipeline (Standard)</option>
                <option value="vlm-auto-engine">VLM Auto (Visual)</option>
                <option value="hybrid-auto-engine">Hybrid (High Prec.)</option>
              </optgroup>
              <optgroup :label="$t('task.groupPaddleOCR')">
                <option value="paddleocr-vl">PaddleOCR-VL</option>
                <option value="paddleocr-vl-vllm">PaddleOCR-VL-VLLM</option>
              </optgroup>
              <optgroup :label="$t('task.groupAudioVideo')">
                <option value="sensevoice">SenseVoice (Audio)</option>
                <option value="video">Video Processing</option>
              </optgroup>
              <optgroup :label="$t('task.groupProfessional')">
                <option value="fasta">FASTA</option>
                <option value="genbank">GenBank</option>
              </optgroup>
            </select>
            <Server class="absolute right-2.5 top-2.5 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
        </div>

        <div class="sm:col-span-2">
          <label class="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">{{ $t('common.search') }}</label>
          <div class="relative">
            <input
              v-model="filters.search"
              @change="applyFilters" 
              type="text"
              :placeholder="$t('common.search') + ' (' + $t('task.fileName') + ' / ID)'"
              class="w-full pl-9 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors text-sm"
            >
            <Search class="absolute left-3 top-2.5 w-4 h-4 text-gray-400 pointer-events-none" />
          </div>
        </div>
      </div>
    </div>

    <div class="card shadow-sm border-gray-100 overflow-hidden">
      
      <div v-if="selectedTasks.length > 0" class="bg-blue-50 px-6 py-2 border-b border-blue-100 flex items-center justify-between transition-all animate-fade-in">
        <div class="flex items-center text-blue-800 text-sm font-medium">
          <CheckSquare class="w-4 h-4 mr-2" />
          {{ $t('common.selected') }} {{ selectedTasks.length }} {{ $t('common.items') }}
        </div>
        <div class="flex gap-2">
          <button
            @click="batchCancel"
            class="text-red-600 hover:text-red-700 hover:bg-red-100 px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center"
          >
            <XCircle class="w-4 h-4 mr-1.5" />
            {{ $t('task.batchCancel') }}
          </button>
          <button
            @click="selectedTasks = []"
            class="text-gray-500 hover:text-gray-700 hover:bg-white px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
          >
            {{ $t('common.deselect') }}
          </button>
        </div>
      </div>

      <div v-if="loading && tasks.length === 0" class="flex flex-col items-center justify-center py-20">
        <LoadingSpinner size="lg" :text="$t('common.loading')" />
      </div>

      <div v-else-if="tasks.length === 0" class="flex flex-col items-center justify-center py-20 text-gray-500">
        <div class="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-4">
          <FileQuestion class="w-8 h-8 text-gray-400" />
        </div>
        <p class="text-lg font-medium text-gray-900">{{ $t('task.noTasks') }}</p>
        <p class="text-sm mt-1">{{ $t('task.noMatchingTasks') }}</p>
        <button @click="clearFilters" class="mt-4 text-primary-600 hover:text-primary-700 text-sm font-medium hover:underline">
          {{ $t('task.clearFilters') }}
        </button>
      </div>

      <div v-else class="overflow-x-auto custom-scrollbar">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50/50">
            <tr>
              <th scope="col" class="px-6 py-3 text-left w-10">
                <input
                  v-model="selectAll"
                  @change="toggleSelectAll"
                  type="checkbox"
                  class="rounded border-gray-300 text-primary-600 focus:ring-primary-500 w-4 h-4 cursor-pointer"
                />
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                {{ $t('task.fileName') }}
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                {{ $t('task.status') }}
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                {{ $t('task.backend') }}
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                {{ $t('task.basicInfo') }}
              </th>
              <th scope="col" class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                {{ $t('task.actions') }}
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-100">
            <tr
              v-for="task in tasks"
              :key="task.task_id"
              :class="{'bg-blue-50/30': selectedTasks.includes(task.task_id)}"
              class="hover:bg-gray-50/80 transition-colors group"
            >
              <td class="px-6 py-4 whitespace-nowrap">
                <input
                  v-model="selectedTasks"
                  :value="task.task_id"
                  type="checkbox"
                  class="rounded border-gray-300 text-primary-600 focus:ring-primary-500 w-4 h-4 cursor-pointer"
                />
              </td>
              <td class="px-6 py-4">
                <div class="flex items-start">
                  <div class="p-2 bg-gray-100 rounded-lg mr-3 group-hover:bg-white group-hover:shadow-sm transition-all">
                    <FileText class="w-5 h-5 text-gray-500" />
                  </div>
                  <div class="min-w-0">
                    <div class="text-sm font-medium text-gray-900 truncate max-w-[200px] sm:max-w-[300px]" :title="task.file_name">
                      {{ task.file_name }}
                    </div>
                    <div class="text-xs text-gray-400 font-mono mt-0.5 flex items-center">
                      {{ task.task_id }}
                      <button @click="copyToClipboard(task.task_id)" class="ml-1.5 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-primary-600" :title="$t('task.copyTaskId')">
                        <Copy class="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <StatusBadge :status="task.status" />
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border border-gray-200">
                  {{ formatBackendName(task.backend) }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                <div class="flex flex-col">
                  <span>{{ formatRelativeTime(task.created_at) }}</span>
                  <span v-if="task.completed_at" class="text-xs text-gray-400 mt-0.5">
                    {{ $t('common.duration') }}: {{ formatDuration(task.created_at, task.completed_at) }}
                  </span>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <div class="flex items-center justify-end gap-2 opacity-60 group-hover:opacity-100 transition-opacity">
                  <router-link
                    :to="`/tasks/${task.task_id}`"
                    class="text-gray-500 hover:text-primary-600 transition-colors p-1.5 rounded hover:bg-primary-50"
                    :title="$t('task.viewDetail')"
                  >
                    <Eye class="w-4 h-4" />
                  </router-link>
                  
                  <button
                    v-if="task.status === 'pending'"
                    @click="cancelTask(task.task_id)"
                    class="text-gray-500 hover:text-red-600 transition-colors p-1.5 rounded hover:bg-red-50"
                    :title="$t('task.cancelTask')"
                  >
                    <XCircle class="w-4 h-4" />
                  </button>
                  <span v-else class="w-7"></span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="total > 0" class="bg-gray-50 px-6 py-4 border-t border-gray-200 flex items-center justify-between">
        <div class="text-sm text-gray-500 hidden sm:block">
           {{ $t('common.pagination', { start: (currentPage - 1) * pageSize + 1, end: Math.min(currentPage * pageSize, total), total: total }) }}
        </div>
        <div class="flex gap-2 w-full sm:w-auto justify-between sm:justify-end">
          <button
            @click="changePage(currentPage - 1)"
            :disabled="currentPage === 1"
            class="p-2 bg-white border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
          >
            <ChevronLeft class="w-4 h-4" />
          </button>
          <div class="flex items-center px-4 bg-white border border-gray-200 rounded-md shadow-sm">
            <span class="text-sm font-medium text-gray-700">{{ currentPage }}</span>
            <span class="text-sm text-gray-400 mx-2">/</span>
            <span class="text-sm text-gray-500">{{ totalPages }}</span>
          </div>
          <button
            @click="changePage(currentPage + 1)"
            :disabled="currentPage === totalPages"
            class="p-2 bg-white border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
          >
            <ChevronRight class="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>

    <ConfirmDialog
      v-model="showCancelDialog"
      :title="$t('common.confirm')"
      :message="cancelDialogMessage"
      @confirm="confirmCancel"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useTaskStore } from '@/stores'
import { useI18n } from 'vue-i18n'
import { formatRelativeTime, formatBackendName, formatDuration } from '@/utils/format'
import StatusBadge from '@/components/StatusBadge.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import {
  Search, RefreshCw, Plus, FileText, Eye, FileQuestion,
  ChevronLeft, ChevronRight, Filter, Server, CheckSquare,
  XCircle, Copy
} from 'lucide-vue-next'
import type { TaskStatus, Backend } from '@/api/types'

const { t } = useI18n()
const taskStore = useTaskStore()

// 使用 store 中的状态
const tasks = computed(() => taskStore.tasks)
const total = computed(() => taskStore.total) // 从 store 获取总数
const loading = ref(false)

const autoRefresh = ref(false)
let refreshInterval: number | null = null

const filters = ref({
  status: '' as TaskStatus | '',
  backend: '' as Backend | '',
  search: '',
})

const pageSize = 20
const currentPage = ref(1)
const totalPages = computed(() => Math.ceil(total.value / pageSize) || 1)

// 批量选择
const selectedTasks = ref<string[]>([])
const selectAll = ref(false)

function toggleSelectAll() {
  if (selectAll.value) {
    // 仅选中当前页的任务
    selectedTasks.value = tasks.value.map(t => t.task_id)
  } else {
    selectedTasks.value = []
  }
}

watch(currentPage, () => {
  selectAll.value = false
  selectedTasks.value = [] // 换页时清除选中
})

// 自动刷新逻辑 (持久化)
watch(autoRefresh, (newVal) => {
  localStorage.setItem('task_list_auto_refresh', String(newVal))
  if (newVal) {
    refreshTasks(false)
    if (!refreshInterval) {
        refreshInterval = window.setInterval(() => refreshTasks(false), 5000)
    }
  } else {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }
})

// 核心刷新函数 (对接服务端分页)
async function refreshTasks(forceLoading = false) {
  if (forceLoading) loading.value = true
  
  try {
    await taskStore.fetchTasks({
      page: currentPage.value,
      page_size: pageSize,
      status: filters.value.status || undefined,
      backend: filters.value.backend || undefined,
      search: filters.value.search || undefined
    })
  } finally {
    if (forceLoading) loading.value = false
  }
}

// 筛选变更
function applyFilters() {
  currentPage.value = 1
  refreshTasks(true)
}

function clearFilters() {
  filters.value.status = ''
  filters.value.backend = ''
  filters.value.search = ''
  applyFilters()
}

// 翻页
function changePage(page: number) {
    if (page >= 1 && page <= totalPages.value) {
        currentPage.value = page
        refreshTasks(true)
    }
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text)
}

// 恢复自动刷新状态
onMounted(async () => {
  await refreshTasks(true)
  
  const savedAutoRefresh = localStorage.getItem('task_list_auto_refresh')
  if (savedAutoRefresh === 'true') {
    autoRefresh.value = true
  }
})

onUnmounted(() => {
  if (refreshInterval) clearInterval(refreshInterval)
})

// 取消任务逻辑
const showCancelDialog = ref(false)
const cancelDialogMessage = ref('')
const taskToCancel = ref<string | string[]>('')

async function cancelTask(taskId: string) {
  taskToCancel.value = taskId
  cancelDialogMessage.value = t('task.confirmCancel')
  showCancelDialog.value = true
}

async function batchCancel() {
  // 注意：这里只能批量取消当前页选中的 pending 任务
  // 实际上用户只能选中当前页的任务，所以逻辑是闭环的
  const pendingTasks = selectedTasks.value.filter(id => {
    const task = tasks.value.find(t => t.task_id === id)
    return task?.status === 'pending'
  })

  if (pendingTasks.length === 0) {
    alert(t('task.noPendingTasksToCancel'))
    return
  }

  taskToCancel.value = pendingTasks
  cancelDialogMessage.value = t('task.confirmBatchCancel', { count: pendingTasks.length })
  showCancelDialog.value = true
}

async function confirmCancel() {
  const ids = Array.isArray(taskToCancel.value) ? taskToCancel.value : [taskToCancel.value]
  for (const id of ids) {
    try {
      await taskStore.cancelTask(id)
    } catch (err) {
      console.error(err)
    }
  }
  selectedTasks.value = []
  selectAll.value = false
  await refreshTasks(true)
}
</script>

<style scoped>
.btn-sm { @apply px-3 py-1.5 text-sm; }
.animate-fade-in { animation: fadeIn 0.4s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.custom-scrollbar::-webkit-scrollbar { height: 6px; width: 6px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f8f9fa; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
.custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
</style>
