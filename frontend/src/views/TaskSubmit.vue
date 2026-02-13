<template>
  <div class="max-w-5xl mx-auto px-4 py-6 animate-fade-in">
    <div class="mb-6 flex justify-between items-end">
      <div>
        <h1 class="text-2xl font-bold text-gray-900 tracking-tight">{{ $t('task.submitTask') }}</h1>
        <p class="mt-1 text-sm text-gray-500">{{ $t('task.processingOptions') }}</p>
      </div>
      <button 
        @click="resetConfig" 
        class="text-xs text-gray-500 hover:text-primary-600 underline transition-colors flex items-center"
        title="恢复默认设置"
      >
        <RotateCcw class="w-3 h-3 mr-1" />
        {{ $t('common.reset') || '重置配置' }}
      </button>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      <div class="lg:col-span-5 order-2 lg:order-1">
        <div class="card shadow-sm hover:shadow-md transition-shadow duration-200 h-full flex flex-col">
          <div class="p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg">
            <h2 class="text-lg font-semibold text-gray-800 flex items-center">
              <Upload class="w-5 h-5 mr-2 text-primary-600" />
              {{ $t('task.selectFile') }}
            </h2>
          </div>
          <div class="p-6 flex-1 flex flex-col">
            <FileUploader
              ref="fileUploader"
              :multiple="true"
              :acceptHint="$t('task.supportedFormatsHint')"
              @update:files="onFilesChange"
              class="flex-1"
            />
            
            <div class="mt-6 pt-6 border-t border-gray-100">
              <button
                @click="submitTasks"
                :disabled="files.length === 0 || submitting"
                class="w-full btn btn-primary py-3 text-base font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-sm hover:shadow-md transition-all transform active:scale-[0.98]"
              >
                <Loader v-if="submitting" class="w-5 h-5 mr-2 animate-spin" />
                <Send v-else class="w-5 h-5 mr-2" />
                {{ submitting ? $t('common.loading') : `${$t('task.submitTask')} (${files.length})` }}
              </button>
              <p v-if="files.length === 0" class="text-center text-xs text-gray-400 mt-2">
                请先选择文件
              </p>
            </div>
          </div>
        </div>
      </div>

      <div class="lg:col-span-7 order-1 lg:order-2">
        <div class="card shadow-sm hover:shadow-md transition-shadow duration-200 bg-white">
          <div class="p-4 border-b border-gray-100 bg-gray-50 rounded-t-lg flex justify-between items-center">
            <h2 class="text-lg font-semibold text-gray-800 flex items-center">
              <Settings class="w-5 h-5 mr-2 text-primary-600" />
              {{ $t('task.processingOptions') }}
            </h2>
            <span class="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded border border-blue-100 font-mono">
              {{ config.backend }}
            </span>
          </div>
          
          <div class="p-6 space-y-6">
            
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
               <button 
                 v-for="preset in presets" 
                 :key="preset.id"
                 @click="applyPreset(preset)"
                 :class="['p-2 rounded-lg border text-left transition-all relative overflow-hidden group', 
                          currentPreset === preset.id ? 'border-primary-500 bg-primary-50 ring-1 ring-primary-500' : 'border-gray-200 hover:border-primary-300 hover:bg-gray-50']"
               >
                 <div class="text-xl mb-1">{{ preset.icon }}</div>
                 <div class="text-xs font-bold text-gray-800">{{ preset.name }}</div>
                 <div class="text-[10px] text-gray-500 leading-tight mt-0.5">{{ preset.desc }}</div>
                 <div v-if="currentPreset === preset.id" class="absolute top-1 right-1 text-primary-600">
                    <CheckCircle class="w-3 h-3" />
                 </div>
               </button>
            </div>

            <hr class="border-gray-100" />

            <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div class="col-span-1 md:col-span-2">
                <label class="block text-sm font-bold text-gray-800 mb-1.5">解析后端 (Backend Engine)</label>
                <div class="relative">
                  <select
                    v-model="config.backend"
                    @change="onBackendChange"
                    class="w-full pl-3 pr-8 py-3 bg-white border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors text-sm font-medium shadow-sm"
                  >
                    <option value="auto">Auto (自动选择)</option>
                    <optgroup label="MinerU Documents">
                      <option value="vlm-auto-engine">MinerU VLM (多模态大模型高精度)</option>
                      <option value="hybrid-auto-engine">MinerU Hybrid (高精度混合解析)</option>
                      <option value="pipeline">Standard Pipeline (通用管道)</option>
                    </optgroup>
                    <optgroup label="Audio / Video / OCR">
                      <option value="paddleocr-vl">PaddleOCR-VL</option>
                      <option value="paddleocr-vl-vllm">PaddleOCR-VL-VLLM</option>
                      <option value="sensevoice">SenseVoice (Audio)</option>
                      <option value="video">Video Processing</option>
                    </optgroup>
                    <optgroup label="Professional Formats">
                      <option value="fasta">FASTA</option>
                      <option value="genbank">GenBank</option>
                    </optgroup>
                  </select>
                  <div class="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                    <ChevronDown class="w-4 h-4 text-gray-500" />
                  </div>
                </div>
                <p class="mt-2 text-xs text-gray-600 bg-gray-50 p-2 rounded border border-gray-100 flex items-start">
                  <Info class="w-3.5 h-3.5 mr-1.5 mt-0.5 text-primary-500 flex-shrink-0" />
                  {{ backendDescription }}
                </p>
              </div>

              <div v-if="showLanguageOption" class="col-span-1">
                <label class="block text-sm font-medium text-gray-700 mb-1.5">{{ $t('task.language') }}</label>
                <select v-model="config.lang" class="w-full px-3 py-2.5 bg-white border border-gray-300 rounded-lg text-sm">
                  <option value="auto">{{ $t('task.langAuto') }}</option>
                  <option value="ch">{{ $t('task.langChinese') }}</option>
                  <option value="en">{{ $t('task.langEnglish') }}</option>
                  <option value="korean">{{ $t('task.langKorean') }}</option>
                  <option value="japan">{{ $t('task.langJapanese') }}</option>
                </select>
              </div>

              <div v-if="showLanguageOption" class="col-span-1">
                <label class="block text-sm font-medium text-gray-700 mb-1.5">{{ $t('task.priorityLabel') }}</label>
                <input v-model.number="config.priority" type="number" min="0" max="100" class="w-full px-3 py-2.5 border border-gray-300 rounded-lg text-sm" />
              </div>
            </div>

            <div v-if="isMinerUBackend" class="pt-4 border-t border-gray-100">
              <h3 class="text-sm font-bold text-gray-800 mb-3 flex items-center">
                <ScanText class="w-4 h-4 mr-2 text-primary-600"/> 识别与解析控制
              </h3>
              
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors">
                  <label class="flex items-center cursor-pointer mb-1">
                    <input v-model="config.table_enable" type="checkbox" class="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500" />
                    <span class="ml-2 text-sm font-medium text-gray-900">启用表格识别</span>
                  </label>
                  <p class="text-xs text-gray-500 pl-6">禁用后，表格将显示为图片。</p>
                </div>

                <div class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors">
                  <label class="flex items-center cursor-pointer mb-1">
                    <input v-model="config.formula_enable" type="checkbox" class="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500" />
                    <span class="ml-2 text-sm font-medium text-gray-900">{{ formulaLabel }}</span>
                  </label>
                  <p class="text-xs text-gray-500 pl-6">禁用后，{{ formulaDescription }}</p>
                </div>

                <div v-if="config.backend !== 'vlm-auto-engine'" class="border border-gray-200 rounded-lg p-3 hover:bg-gray-50 transition-colors md:col-span-2" :class="{'bg-orange-50 border-orange-200': config.force_ocr}">
                  <label class="flex items-center cursor-pointer mb-1">
                    <input v-model="config.force_ocr" type="checkbox" class="w-4 h-4 text-orange-500 rounded border-gray-300 focus:ring-orange-500" />
                    <span class="ml-2 text-sm font-medium text-gray-900">强制启用 OCR <span v-if="config.force_ocr" class="text-orange-600 text-xs ml-2">(已启用)</span></span>
                  </label>
                  <p class="text-xs text-gray-500 pl-6">仅在识别效果极差（如模糊扫描件）时启用，需选择正确的 OCR 语言。</p>
                </div>
              </div>
            </div>

            <div class="bg-gray-50 rounded-lg p-1 border border-gray-200">
              <button @click="showAdvanced = !showAdvanced" class="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-100 transition-colors rounded-lg group">
                <div class="flex items-center text-sm font-medium text-gray-700">
                  <Sliders class="w-4 h-4 mr-2 text-gray-500 group-hover:text-primary-600" />
                  高级配置 <span class="ml-2 text-xs text-gray-400 font-normal">(页码、去水印、Office转换、调试)</span>
                </div>
                <component :is="showAdvanced ? ChevronUp : ChevronDown" class="w-4 h-4 text-gray-400" />
              </button>

              <div v-show="showAdvanced" class="px-4 pb-4 pt-2 space-y-5 animate-fade-in-down">
                <div v-if="isMinerUBackend">
                  <label class="block text-xs font-bold text-gray-600 uppercase tracking-wide mb-2">解析范围 (Page Range)</label>
                  <div class="flex items-center space-x-3 bg-white p-3 rounded border border-gray-200">
                    <div class="flex items-center space-x-2 flex-1">
                      <span class="text-xs text-gray-500">Start:</span>
                      <input v-model.number="config.start_page" type="number" min="0" class="form-input-sm w-full" placeholder="0" />
                    </div>
                    <span class="text-gray-300">|</span>
                    <div class="flex items-center space-x-2 flex-1">
                      <span class="text-xs text-gray-500">End:</span>
                      <input v-model.number="config.end_page" type="number" min="0" class="form-input-sm w-full" placeholder="Auto" />
                    </div>
                  </div>
                  <p class="mt-1 text-[10px] text-gray-400">留空或 -1 表示处理到文件末尾。起始页从 0 开始。</p>
                </div>

                <div v-if="['pipeline', 'paddleocr-vl', 'paddleocr-vl-vllm', 'auto'].includes(config.backend)">
                   <label class="block text-xs font-bold text-gray-600 uppercase tracking-wide mb-2">预处理增强</label>
                   <div class="grid grid-cols-1 gap-3">
                      <div class="border border-gray-200 rounded p-3 bg-white">
                        <label class="flex items-center cursor-pointer mb-2">
                          <input v-model="config.remove_watermark" type="checkbox" class="form-checkbox text-purple-600 rounded border-gray-300 h-4 w-4" />
                          <span class="ml-2 text-sm font-medium text-gray-800">启用去水印 (Watermark Removal)</span>
                        </label>
                        <div v-if="config.remove_watermark" class="pl-6 pt-1 space-y-2 animate-fade-in">
                           <div class="flex items-center justify-between text-xs text-gray-500">
                             <span>检测置信度: <span class="font-mono text-purple-600">{{ config.watermark_conf_threshold }}</span></span>
                           </div>
                           <input v-model.number="config.watermark_conf_threshold" type="range" min="0.1" max="0.9" step="0.05" class="w-full h-1.5 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-purple-600" />
                        </div>
                      </div>

                      <div v-if="['auto', 'pipeline'].includes(config.backend)" class="border border-gray-200 rounded p-3 bg-white">
                         <label class="flex items-center cursor-pointer">
                          <input v-model="config.convert_office_to_pdf" type="checkbox" class="form-checkbox text-primary-600 rounded border-gray-300 h-4 w-4" />
                          <span class="ml-2 text-sm font-medium text-gray-800">Office 转 PDF 深度解析</span>
                        </label>
                        <p class="pl-6 mt-1 text-xs text-gray-500">
                          推荐启用。先转换为 PDF 后再解析，可完整提取 Word/PPT 中的复杂图表和排版。
                        </p>
                      </div>
                   </div>
                </div>

                <div v-if="config.backend === 'video' || config.backend === 'sensevoice'">
                   <label class="block text-xs font-bold text-blue-600 uppercase tracking-wide mb-2">媒体处理参数</label>
                   <div class="bg-blue-50 border border-blue-100 rounded p-3 space-y-2">
                      <div v-if="config.backend === 'video'">
                         <label class="flex items-center cursor-pointer"><input v-model="config.keep_audio" type="checkbox" class="mr-2 rounded text-blue-600"/> <span class="text-sm">保留音频轨道文件</span></label>
                         <label class="flex items-center cursor-pointer mt-2"><input v-model="config.enable_keyframe_ocr" type="checkbox" class="mr-2 rounded text-blue-600"/> <span class="text-sm">启用关键帧 OCR 内容识别</span></label>
                      </div>
                      <div v-if="config.backend === 'sensevoice'">
                         <label class="flex items-center cursor-pointer"><input v-model="config.enable_speaker_diarization" type="checkbox" class="mr-2 rounded text-blue-600"/> <span class="text-sm">启用说话人分离 (Speaker Diarization)</span></label>
                      </div>
                   </div>
                </div>

                <div v-if="isMinerUBackend">
                  <div class="pt-2 border-t border-dashed border-gray-200 mt-2">
                    <label class="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2">DEBUG OUTPUT</label>
                    <div class="flex space-x-6">
                       <label class="flex items-center cursor-pointer text-xs text-gray-500 hover:text-gray-700">
                          <input v-model="config.draw_layout" type="checkbox" class="mr-1.5 rounded border-gray-300" />
                          生成 Layout 标注 PDF
                       </label>
                       <label class="flex items-center cursor-pointer text-xs text-gray-500 hover:text-gray-700">
                          <input v-model="config.draw_span" type="checkbox" class="mr-1.5 rounded border-gray-300" />
                          生成 Span 标注 PDF
                       </label>
                    </div>
                  </div>
                </div>

              </div>
            </div>

          </div>
        </div>

        <div v-if="errorMessage" class="mt-4 rounded-lg bg-red-50 border border-red-200 p-4 animate-fade-in flex items-start shadow-sm">
          <AlertCircle class="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div class="ml-3 flex-1">
            <h3 class="text-sm font-medium text-red-800">{{ $t('common.error') }}</h3>
            <p class="mt-1 text-sm text-red-700">{{ errorMessage }}</p>
          </div>
          <button @click="errorMessage = ''" class="ml-auto -mr-1 -mt-1 p-1 text-red-600 hover:text-red-800 rounded-full hover:bg-red-100 transition-colors">
            <X class="w-4 h-4" />
          </button>
        </div>

        <div v-if="submitProgress.length > 0" class="mt-6 card overflow-hidden animate-fade-in shadow-sm">
          <div class="bg-gray-50 px-4 py-3 border-b border-gray-100 flex justify-between items-center">
            <h3 class="text-sm font-semibold text-gray-900">{{ $t('common.progress') }}</h3>
            <span class="text-xs text-gray-500">{{ submitProgress.filter(p => p.success).length }} / {{ submitProgress.length }} 完成</span>
          </div>
          <div class="max-h-60 overflow-y-auto divide-y divide-gray-100 custom-scrollbar">
            <div
              v-for="(progress, index) in submitProgress"
              :key="index"
              class="flex items-center justify-between p-3 px-4 hover:bg-gray-50 transition-colors"
            >
              <div class="flex items-center flex-1 min-w-0">
                <FileText :class="['w-4 h-4 mr-3 flex-shrink-0', progress.success ? 'text-green-500' : progress.error ? 'text-red-500' : 'text-gray-400']" />
                <div class="truncate">
                   <p class="text-sm text-gray-700 truncate font-medium">{{ progress.fileName }}</p>
                   <p v-if="progress.taskId" class="text-[10px] text-gray-400 font-mono tracking-tight">{{ progress.taskId }}</p>
                   <p v-if="progress.errorMsg" class="text-[10px] text-red-500 truncate">{{ progress.errorMsg }}</p>
                </div>
              </div>
              <div class="flex items-center ml-3">
                <CheckCircle v-if="progress.success" class="w-5 h-5 text-green-500" />
                <XCircle v-else-if="progress.error" class="w-5 h-5 text-red-500" />
                <Loader v-else class="w-5 h-5 text-primary-500 animate-spin" />
              </div>
            </div>
          </div>
          
          <div v-if="!submitting" class="p-3 bg-gray-50 border-t border-gray-100 flex justify-end space-x-3">
            <button @click="resetForm" class="btn btn-secondary btn-sm text-xs">
              {{ $t('common.continue') }} (清空)
            </button>
            <router-link to="/tasks" class="btn btn-primary btn-sm text-xs flex items-center">
              查看任务列表 <ArrowRight class="w-3 h-3 ml-1"/>
            </router-link>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useTaskStore } from '@/stores'
import FileUploader from '@/components/FileUploader.vue'
import {
  Upload, Loader, AlertCircle, X, FileText, CheckCircle, XCircle,
  Settings, ChevronDown, ChevronUp, Info, RotateCcw, ArrowRight,
  ScanText, Send, Sliders, Wand
} from 'lucide-vue-next'
import type { Backend, Language, ParseMethod } from '@/api/types'

const { t } = useI18n()
const router = useRouter()
const taskStore = useTaskStore()

const fileUploader = ref<InstanceType<typeof FileUploader>>()
const files = ref<File[]>([])
const submitting = ref(false)
const errorMessage = ref('')
const showAdvanced = ref(false)
const currentPreset = ref('default')

interface SubmitProgress { fileName: string; success: boolean; error: boolean; taskId?: string; errorMsg?: string }
const submitProgress = ref<SubmitProgress[]>([])

// 预设配置
const presets = [
  { id: 'default', name: '通用文档', desc: '标准 Pipeline，速度快', icon: '📄', config: { backend: 'pipeline', force_ocr: false, formula_enable: true, table_enable: true } },
  { id: 'academic', name: '学术论文', desc: 'Hybrid 模式，公式增强', icon: '🎓', config: { backend: 'hybrid-auto-engine', force_ocr: false, formula_enable: true, table_enable: true } },
  { id: 'scanned', name: '扫描件', desc: 'Hybrid + 强制 OCR', icon: '🖨️', config: { backend: 'hybrid-auto-engine', force_ocr: true, formula_enable: false, table_enable: true } },
  { id: 'complex', name: '复杂图表', desc: 'VLM 视觉模型', icon: '📊', config: { backend: 'vlm-auto-engine', force_ocr: false, formula_enable: true, table_enable: true } },
]

const defaultConfig = {
  backend: 'auto' as Backend,
  lang: 'auto' as Language,
  method: 'auto' as ParseMethod,
  force_ocr: false,
  formula_enable: true,
  table_enable: true,
  priority: 0,
  start_page: undefined as number | undefined,
  end_page: undefined as number | undefined,
  draw_layout: false,
  draw_span: false,
  convert_office_to_pdf: false,
  keep_audio: false,
  enable_keyframe_ocr: false,
  ocr_backend: 'paddleocr-vl',
  keep_keyframes: false,
  enable_speaker_diarization: false,
  remove_watermark: false,
  watermark_conf_threshold: 0.35,
  watermark_dilation: 10,
}

const config = reactive({ ...defaultConfig })

// 预设应用
function applyPreset(preset: any) {
  currentPreset.value = preset.id
  Object.assign(config, preset.config)
  // 如果是扫描件，建议去水印
  if (preset.id === 'scanned') config.remove_watermark = true
}

// 计算属性
const isMinerUBackend = computed(() => ['pipeline', 'vlm-auto-engine', 'hybrid-auto-engine'].includes(config.backend))
const showLanguageOption = computed(() => isMinerUBackend.value || ['paddleocr-vl', 'paddleocr-vl-vllm', 'sensevoice', 'auto'].includes(config.backend))
const backendDescription = computed(() => {
  const map: Record<string, string> = {
    'vlm-auto-engine': '多模态大模型高精度解析，仅支持中英文文档。擅长复杂排版。',
    'hybrid-auto-engine': '高精度混合解析，支持多语言。结合规则与模型，精度最高。',
    'pipeline': '传统多模型管道解析，支持多语言，无幻觉，速度较快。',
    'auto': '自动根据文件类型选择最合适的解析引擎。',
    'paddleocr-vl': '纯 OCR 模式，适合简单图片提取文字。',
  }
  return map[config.backend] || '通用处理引擎'
})
const formulaLabel = computed(() => config.backend === 'vlm-auto-engine' ? '启用行间公式识别' : (config.backend === 'hybrid-auto-engine' ? '启用行内公式识别' : '启用公式识别'))
const formulaDescription = computed(() => config.backend === 'vlm-auto-engine' ? '行间公式将显示为图片' : (config.backend === 'hybrid-auto-engine' ? '行内公式将不会被检测或解析' : '公式将不会被特殊处理'))

function onFilesChange(newFiles: File[]) { files.value = newFiles }
function onBackendChange() { currentPreset.value = 'custom'; if (config.backend === 'vlm-auto-engine') config.lang = 'ch' }

onMounted(() => {
  const saved = localStorage.getItem('task_submit_config')
  if (saved) { try { const p = JSON.parse(saved); Object.assign(config, p); } catch(e){} }
})

watch(() => config, () => { localStorage.setItem('task_submit_config', JSON.stringify(config)) }, { deep: true })

function resetConfig() { Object.assign(config, defaultConfig); localStorage.removeItem('task_submit_config'); currentPreset.value = 'default' }

async function submitTasks() {
  if (files.value.length === 0) { errorMessage.value = t('task.pleaseSelectFile'); return }
  
  // 参数校验
  if (config.start_page !== undefined && config.end_page !== undefined && config.end_page !== -1) {
    if (config.end_page < config.start_page) {
      errorMessage.value = "结束页码不能小于开始页码"
      showAdvanced.value = true
      return
    }
  }

  submitting.value = true; errorMessage.value = ''
  submitProgress.value = files.value.map(f => ({ fileName: f.name, success: false, error: false }))

  for (let i = 0; i < files.value.length; i++) {
    try {
      const submitConfig = { ...config }
      if (submitConfig.force_ocr) submitConfig.method = 'ocr'
      if (submitConfig.end_page === undefined) delete (submitConfig as any).end_page
      const response = await taskStore.submitTask({ file: files.value[i], ...submitConfig })
      submitProgress.value[i].success = true; submitProgress.value[i].taskId = response.task_id
    } catch (err: any) {
      submitProgress.value[i].error = true; submitProgress.value[i].errorMsg = err.message
    }
  }
  submitting.value = false
  if (submitProgress.value.every(p => p.success) && files.value.length === 1) {
    setTimeout(() => { router.push(`/tasks/${submitProgress.value[0].taskId}`) }, 500)
  }
}
</script>

<style scoped>
.form-input-sm { @apply px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-primary-500 transition-colors; }
.form-checkbox { @apply rounded border-gray-300 focus:ring-primary-500 cursor-pointer; }
.animate-fade-in { animation: fadeIn 0.3s ease-out; }
.animate-fade-in-down { animation: fadeInDown 0.2s ease-out; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-track { background: #f8f9fa; }
.custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
