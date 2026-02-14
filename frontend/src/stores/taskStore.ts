/**
 * 任务状态管理
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { taskApi } from '@/api'
import type { Task, SubmitTaskRequest, TaskStatus } from '@/api/types'

export const useTaskStore = defineStore('task', () => {
  // 状态
  const tasks = ref<Task[]>([])
  const currentTask = ref<Task | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // 计算属性
  const pendingTasks = computed(() =>
    tasks.value.filter(t => t.status === 'pending')
  )

  const processingTasks = computed(() =>
    tasks.value.filter(t => t.status === 'processing')
  )

  const completedTasks = computed(() =>
    tasks.value.filter(t => t.status === 'completed')
  )

  const failedTasks = computed(() =>
    tasks.value.filter(t => t.status === 'failed')
  )

  // 动作
  /**
   * 提交任务
   */
  async function submitTask(request: SubmitTaskRequest) {
    loading.value = true
    error.value = null

    try {
      const response = await taskApi.submitTask(request)

      // 添加到任务列表
      const newTask: Task = {
        task_id: response.task_id,
        file_name: response.file_name,
        status: response.status,
        backend: request.backend || 'pipeline', // 默认值回退
        priority: request.priority || 0,
        error_message: null,
        created_at: response.created_at,
        started_at: null,
        completed_at: null,
        worker_id: null,
        retry_count: 0,
        result_path: null,
      }

      tasks.value.unshift(newTask)
      return response
    } catch (err: any) {
      error.value = err.message || '提交任务失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取任务状态
   */
  async function fetchTaskStatus(
    taskId: string,
    uploadImages: boolean = false,
    format: 'markdown' | 'json' | 'both' = 'markdown'
  ) {
    // 只有当当前没有显示该任务，或者强制刷新时才显示 loading
    // 避免轮询时界面闪烁
    if (!currentTask.value || currentTask.value.task_id !== taskId) {
        loading.value = true
    }
    
    error.value = null

    try {
      const response = await taskApi.getTaskStatus(taskId, uploadImages, format)
      
      // 构建完整的 Task 对象
      const updatedTask: Task = {
        task_id: response.task_id,
        file_name: response.file_name,
        status: response.status,
        backend: response.backend,
        priority: response.priority,
        error_message: response.error_message,
        created_at: response.created_at,
        started_at: response.started_at,
        completed_at: response.completed_at,
        worker_id: response.worker_id,
        retry_count: response.retry_count,
        result_path: null, // API 响应中通常没有这个字段，或者叫 result_dir
        data: response.data, // 这里包含了 content, json_content, pdf_path 等
      }

      currentTask.value = updatedTask

      // 更新任务列表中的任务状态，保持列表数据同步
      const index = tasks.value.findIndex(t => t.task_id === taskId)
      if (index !== -1) {
        // 只更新状态和时间信息，避免列表页重绘太重
        tasks.value[index] = {
            ...tasks.value[index],
            status: updatedTask.status,
            completed_at: updatedTask.completed_at,
            started_at: updatedTask.started_at,
            error_message: updatedTask.error_message
        }
      }

      return response
    } catch (err: any) {
      error.value = err.message || '获取任务状态失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * 取消任务
   */
  async function cancelTask(taskId: string) {
    loading.value = true
    error.value = null

    try {
      await taskApi.cancelTask(taskId)

      // 更新任务状态
      const task = tasks.value.find(t => t.task_id === taskId)
      if (task) {
        task.status = 'cancelled'
      }

      if (currentTask.value?.task_id === taskId) {
        currentTask.value.status = 'cancelled'
      }
    } catch (err: any) {
      error.value = err.message || '取消任务失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取任务列表
   */
  async function fetchTasks(status?: TaskStatus, limit: number = 100) {
    loading.value = true
    error.value = null

    try {
      const response = await taskApi.listTasks(status, limit)
      tasks.value = response.tasks
      return response
    } catch (err: any) {
      error.value = err.message || '获取任务列表失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * 轮询任务状态
   */
  function pollTaskStatus(
    taskId: string,
    interval: number = 2000,
    onUpdate?: (task: Task) => void,
    format: 'markdown' | 'json' | 'both' = 'markdown'
  ): () => void {
    let timerId: number | null = null
    let stopped = false

    const poll = async () => {
      if (stopped) return

      try {
        // 调用 fetchTaskStatus 更新 store
        await fetchTaskStatus(taskId, false, format)
        
        if (currentTask.value && onUpdate) {
            onUpdate(currentTask.value)
        }

        const status = currentTask.value?.status
        // 如果任务完成或失败，停止轮询
        if (status === 'completed' || status === 'failed' || status === 'cancelled') {
          stopped = true
          return
        }

        // 继续轮询
        if (!stopped) {
          timerId = window.setTimeout(poll, interval)
        }
      } catch (err) {
        console.error('轮询任务状态失败:', err)
        // 发生错误时可以选择继续重试几次，或者停止
        // 这里选择停止以防止无限错误循环
        stopped = true
      }
    }

    // 开始轮询
    poll()

    // 返回停止函数
    return () => {
      stopped = true
      if (timerId) {
        clearTimeout(timerId)
        timerId = null
      }
    }
  }

  /**
   * 清空错误
   */
  function clearError() {
    error.value = null
  }

  /**
   * 重置状态
   */
  function reset() {
    tasks.value = []
    currentTask.value = null
    loading.value = false
    error.value = null
  }

  return {
    // 状态
    tasks,
    currentTask,
    loading,
    error,

    // 计算属性
    pendingTasks,
    processingTasks,
    completedTasks,
    failedTasks,

    // 动作
    submitTask,
    fetchTaskStatus,
    cancelTask,
    fetchTasks,
    pollTaskStatus,
    clearError,
    reset,
  }
})
