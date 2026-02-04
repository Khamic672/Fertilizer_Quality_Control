<template>
  <div class="page">
    <header class="topbar">
        <div class="brand">
          <div class="brand-mark">QC</div>
          <div>
            <p class="brand-title">Fertilizer Quality Control</p>
          </div>
        </div>
      <div class="top-actions">
        <span :class="['pill', backendStatusClass]">Backend: {{ backendLabel }}</span>
      </div>
    </header>

    <main class="layout">
      <div class="hero-heading">
        <h2>ระบบควบคุมคุณภาพปุ๋ย</h2>
        <p class="muted">ถ่ายภาพปุ๋ยเพื่อตรวจสอบคุณภาพด้วยการวิเคราะห์ภาพ AI</p>
      </div>

      <section id="upload-card" class="card upload-card">
        <div class="upload-left">
            <div
            class="drop-area"
            :class="{ 'drop-area--filled': hasSelection || cameraActive }"
            @click="handleDropAreaClick"
            @dragover.prevent
            @drop.prevent="onDrop"
          >
            <input ref="fileInput" type="file" accept="image/*" multiple hidden @change="onFileChange" />
            <div v-if="!hasSelection && !cameraActive" class="drop-inner">
              <div class="camera-icon">📷</div>
              <p class="muted center">ถ่ายรูป หรืออัปโหลดหลายรูปในล็อตเดียว</p>
            </div>
            <div v-if="!hasSelection && !cameraActive" class="upload-buttons">
              <button class="ghost pill-button" @click.stop="startCamera">📷 ถ่ายรูป</button>
              <button class="ghost pill-button" @click.stop="triggerFileSelect">⬆️ อัปโหลดรูป</button>
            </div>
            <div v-if="cameraActive && !hasSelection" class="camera-preview">
              <video ref="videoRef" autoplay playsinline muted></video>
              <div class="camera-controls">
                <button class="primary" @click.stop="capturePhoto">📸 ถ่ายภาพ</button>
                <button class="ghost" @click.stop="stopCamera">ปิดกล้อง</button>
              </div>
              <p v-if="cameraError" class="alert error small">{{ cameraError }}</p>
            </div>
            <p v-else-if="cameraError && !hasSelection" class="alert error small">{{ cameraError }}</p>
            <div v-if="hasSelection" class="carousel" :class="{ 'carousel--solo': selectedFiles.length < 2 }">
              <button class="nav-btn nav-btn--left" v-if="selectedFiles.length > 1" @click.stop="prevSelected">&lt;</button>
              <div v-if="activeSelection" class="preview-thumb">
                <img :src="activeSelection.preview" :alt="activeSelection.file.name" />
                <p class="thumb-name">{{ activeSelection.file.name }}</p>
              </div>
              <button class="nav-btn nav-btn--right" v-if="selectedFiles.length > 1" @click.stop="nextSelected">&gt;</button>
            </div>
          </div>

          <div class="input-card" v-if="selectedFiles.length">
            <p class="section-label">รูปที่เลือก ({{ selectedFiles.length }})</p>
            <ul class="muted">
              <li v-for="fileItem in selectedFiles" :key="fileItem.file.name + fileItem.file.size">
                {{ fileItem.file.name }} — {{ Math.round(fileItem.file.size / 1024) }} KB
              </li>
            </ul>
          </div>

          <div class="input-card">
            <p class="section-label">สูตรปุ๋ยที่ต้องการวิเคราะห์</p>
            <div class="input-grid">
              <div class="field">
                <label>สูตรปุ๋ย (NPK)</label>
                <input v-model="formInputs.formula" placeholder="เช่น 15-15-15" />
              </div>
              <div class="field">
                <label>Lot number</label>
                <input v-model="formInputs.lotNumber" placeholder="ระบุ" />
              </div>
              <div class="field">
                <label>Threshold (% error)*</label>
                <input v-model.number="formInputs.threshold" type="number" min="0" max="100" placeholder="ระบุ" />
              </div>
            </div>
          </div>

          <div class="action-row action-row--split">
            <button class="ghost" :disabled="!hasSelection" @click="resetSelection">เลือกรูปใหม่</button>
            <button class="primary" :disabled="loading || !hasSelection" @click="processSelected">วิเคราะห์คุณภาพ</button>
          </div>

          <div v-if="error" class="alert error">
            <p>{{ error }}</p>
          </div>
        </div>

        <div class="upload-results" v-if="results">
          <template v-if="isBatchResult">
            <div class="input-card">
              <p class="section-label">สรุปล็อตเดียวกัน</p>
              <p class="muted">
                Lot: {{ batchSummary?.lot_number || '—' }} • สูตร: {{ batchSummary?.formula || '—' }} • Threshold:
                {{ batchSummary?.threshold ?? '—' }}
              </p>
              <div class="summary-row">
                <div>จำนวนรูป: {{ batchSummary?.total_images || 0 }}</div>
                <div>ผ่าน: {{ batchSummary?.passed_images || 0 }}</div>
                <div>
                  <span :class="['status-pill', batchSummary?.status || 'ok']">{{ batchSummary?.status || 'ok' }}</span>
                </div>
              </div>
            </div>

            <div class="carousel" v-if="activeResult">
              <button class="nav-btn nav-btn--left" v-if="batchItems.length > 1" @click="prevItem">‹</button>
              <div class="preview-card">
                <p class="section-label">{{ activeResult.filename || `ภาพที่ ${activeIndex + 1}` }}</p>
                <img :src="previewImage" alt="preview" />
                <div v-if="hasSegmentation" class="mask-legend">
                  <p class="mask-legend__title">Legend</p>
                  <div class="mask-legend__items">
                    <div v-for="item in maskLegend" :key="item.label" class="mask-legend__item">
                      <span class="mask-legend__swatch" :style="{ backgroundColor: item.color }"></span>
                      <span>{{ item.label }}</span>
                    </div>
                  </div>
                </div>
              </div>
              <button class="nav-btn nav-btn--right" v-if="batchItems.length > 1" @click="nextItem">›</button>
            </div>

            <div class="status-bars">
              <div class="bar-row" v-for="metric in metrics" :key="metric.key">
                <div class="bar-label">
                  <span>{{ metric.label }}</span>
                  <span>{{ metric.value.toFixed(2) }}</span>
                </div>
                <div class="bar-track">
                  <div
                    class="bar-fill"
                    :style="{
                      width: metric.percent + '%',
                      backgroundColor: metric.color
                    }"
                  />
                </div>
              </div>
            </div>

            <div class="checklist mini" :class="statusTone(activeResult?.status_level)">
              <p class="checklist-title">
                <span>สถานะ</span>
              </p>
              <ul>
                <li v-for="status in statusList" :key="status.message" :class="status.level">
                  <span class="dot" />
                  <span>{{ status.message }}</span>
                </li>
              </ul>
            </div>
          </template>

          <template v-else>
            <div class="preview-card">
              <p class="section-label">Preview</p>
              <img :src="previewImage" alt="preview" />
              <div v-if="hasSegmentation" class="mask-legend">
                <p class="mask-legend__title">Legend</p>
                <div class="mask-legend__items">
                  <div v-for="item in maskLegend" :key="item.label" class="mask-legend__item">
                    <span class="mask-legend__swatch" :style="{ backgroundColor: item.color }"></span>
                    <span>{{ item.label }}</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="status-bars">
              <div class="bar-row" v-for="metric in metrics" :key="metric.key">
                <div class="bar-label">
                  <span>{{ metric.label }}</span>
                  <span>{{ metric.value.toFixed(2) }}</span>
                </div>
                <div class="bar-track">
                  <div
                    class="bar-fill"
                    :style="{
                      width: metric.percent + '%',
                      backgroundColor: metric.color
                    }"
                  />
                </div>
              </div>
            </div>
            <div class="checklist" :class="checklistTone">
              <p class="checklist-title">
                <span>{{ checklistTone === 'bad' ? 'Warnings' : 'Checklist' }}</span>
                <button class="ghost small" @click="acknowledge">OK</button>
              </p>
              <ul>
                <li v-for="item in statusList" :key="item.message" :class="item.level">
                  <span class="dot" />
                  <span>{{ item.message }}</span>
                </li>
              </ul>
            </div>
          </template>
        </div>

        <div v-else class="upload-results placeholder">
          <p class="muted">ผลลัพธ์จะแสดงที่นี่เมื่อประมวลผลเสร็จ</p>
        </div>
      </section>

      <section class="card history-card">
        <div class="history-head">
          <h3>ประวัติการวิเคราะห์</h3>
          <button class="ghost icon-button" @click="openExportModal">
            ↓ ดาวน์โหลด Excel
          </button>
        </div>

        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>สูตร</th>
                <th>Lot number</th>
                <th>Treshold</th>
                <th>จำนวนรูปทั้งหมด</th>
                <th>จำนวนรูปที่ผ่าน</th>
                <th>วันที่ตรวจสอบ</th>
                <th>ผลตรวจ</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in filteredHistory" :key="row.id">
                <td>{{ row.formula || '—' }}</td>
                <td>{{ row.lot_number || row.name }}</td>
                <td>{{ row.threshold != null ? row.threshold + '%' : '—' }}</td>
                <td>{{ row.total_images ?? 1 }}</td>
                <td>{{ row.passed_images ?? (row.status === 'ok' ? 1 : 0) }}</td>
                <td>{{ row.date }}</td>
                <td>
                  <span :class="['status-pill', row.status]">{{ row.status }}</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>

    <div v-if="showExportModal" class="modal-backdrop">
      <div class="modal-card">
        <div class="modal-head">
          <p class="modal-title">ดาวน์โหลดข้อมูลรายการเป็นไฟล์ Excel</p>
          <button class="ghost small" @click="closeExportModal">✕</button>
        </div>
        <div class="modal-body">
          <label class="modal-label">ระยะเวลาของข้อมูล*</label>
          <div class="modal-row">
            <div class="field">
              <label>เริ่ม</label>
              <input v-model="exportRange.start" type="date" />
            </div>
            <div class="field">
              <label>สิ้นสุด</label>
              <input v-model="exportRange.end" type="date" />
            </div>
          </div>
          <p v-if="exportError" class="alert error small">{{ exportError }}</p>
        </div>
        <div class="modal-actions">
          <button class="ghost" @click="closeExportModal" :disabled="exporting">ยกเลิก</button>
          <button class="primary" @click="downloadHistory" :disabled="exporting">
            {{ exporting ? 'กำลังดาวน์โหลด...' : 'ยืนยัน' }}
          </button>
        </div>
      </div>
    </div>

    <LoadingSpinner v-if="loading" />
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import LoadingSpinner from './components/LoadingSpinner.vue'

const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000/api'

const fileInput = ref(null)
const loading = ref(false)
const error = ref('')
const exporting = ref(false)
const results = ref(null)
const selectedFiles = ref([])
const history = ref([])
const backendStatus = ref('checking')
const cameraActive = ref(false)
const cameraError = ref('')
const videoRef = ref(null)
const formInputs = ref({
  formula: '',
  lotNumber: '',
  threshold: 5
})
const showExportModal = ref(false)
const exportError = ref('')
const exportRange = ref({
  start: '',
  end: ''
})

const filters = ref({
  lot: '',
  status: ''
})

const maskLegend = [
  { label: 'Black DAP', color: '#2D2A32' },
  { label: 'Red MOP', color: '#E11D48' },
  { label: 'White AMP', color: '#38BDF8' },
  { label: 'White Boron', color: '#F472B6' },
  { label: 'White Mg', color: '#22C55E' },
  { label: 'Yellow Urea', color: '#F59E0B' }
]

const isBatchResult = computed(() => results.value?.mode === 'batch')
const batchItems = computed(() => (isBatchResult.value ? results.value?.items || [] : []))
const activeIndex = ref(0)
const selectedPreviewIndex = ref(0)
const activeResult = computed(() => {
  if (isBatchResult.value) {
    return batchItems.value[activeIndex.value] || batchItems.value[0] || null
  }
  return results.value
})

const activeSelection = computed(() => selectedFiles.value[selectedPreviewIndex.value] || selectedFiles.value[0] || null)

const hasSegmentation = computed(() => Boolean(activeResult.value?.segmentation))
const previewImage = computed(() => {
  const result = activeResult.value
  if (!result) return ''
  return result.segmentation || result.original || ''
})

const batchSummary = computed(() => {
  if (isBatchResult.value) return results.value?.summary || null
  if (!results.value) return null
  const level = results.value.status_level || 'ok'
  return {
    total_images: 1,
    passed_images: results.value.passed ? 1 : level === 'bad' ? 0 : 1,
    status: level,
    formula: results.value.inputs?.formula,
    lot_number: results.value.inputs?.lot_number,
    threshold: results.value.inputs?.threshold
  }
})

const hasSelection = computed(() => selectedFiles.value.length > 0)

const clearSelectedFiles = () => {
  selectedFiles.value.forEach((item) => URL.revokeObjectURL(item.preview))
  selectedFiles.value = []
  selectedPreviewIndex.value = 0
}

const resetSelection = () => {
  stopCamera()
  clearSelectedFiles()
  results.value = null
  error.value = ''
  cameraError.value = ''
}

const buildMetrics = (npk, targetNpk, npkErrors, thresholdPercent) => {
  const values = npk || { N: 0, P: 0, K: 0 }
  const threshold = thresholdPercent ?? 5

  const colorFor = (key) => {
    const diff = npkErrors && typeof npkErrors[key] === 'number' ? npkErrors[key] : null
    if (diff === null) return '#1f8f45'
    return diff <= threshold ? '#1f8f45' : '#c0392b'
  }

  const total = 30
  return [
    { key: 'n', label: 'N', value: values.N, percent: Math.min(100, (values.N / total) * 100), color: colorFor('N') },
    { key: 'p', label: 'P', value: values.P, percent: Math.min(100, (values.P / total) * 100), color: colorFor('P') },
    { key: 'k', label: 'K', value: values.K, percent: Math.min(100, (values.K / total) * 100), color: colorFor('K') }
  ]
}

const statusTone = (level) => {
  if (level === 'bad') return 'bad'
  if (level === 'warn') return 'warn'
  return 'good'
}

const metrics = computed(() =>
  buildMetrics(
    activeResult.value?.npk,
    activeResult.value?.target_npk || results.value?.inputs?.target_npk,
    activeResult.value?.npk_errors,
    results.value?.inputs?.threshold ?? results.value?.summary?.threshold
  )
)

const statusList = computed(() => {
  const result = activeResult.value
  if (!result) return []
  if (Array.isArray(result.status)) return result.status
  if (result.status && typeof result.status === 'object') {
    const { level, message } = result.status
    if (message) return [{ level: level || result.status_level || 'ok', message }]
  }
  if (result.status_message) {
    return [{ level: result.status_level || 'ok', message: result.status_message }]
  }
  return []
})
const checklistTone = computed(() => {
  if (!statusList.value.length) return 'neutral'
  if (statusList.value.some((s) => s.level === 'bad')) return 'bad'
  if (statusList.value.some((s) => s.level === 'warn')) return 'warn'
  return 'good'
})

const backendLabel = computed(() => {
  if (backendStatus.value === 'ok') return 'online'
  if (backendStatus.value === 'error') return 'offline'
  return 'checking...'
})

const backendStatusClass = computed(() => {
  if (backendStatus.value === 'ok') return 'pill-success'
  if (backendStatus.value === 'error') return 'pill-error'
  return 'pill-warn'
})

const filteredHistory = computed(() =>
  history.value.filter((item) => {
    if (filters.value.status && item.status !== filters.value.status) return false
    if (filters.value.lot && item.lot_number !== filters.value.lot) return false
    return true
  })
)

const triggerFileSelect = () => {
  if (fileInput.value) fileInput.value.click()
}

const handleDropAreaClick = () => {
  if (hasSelection.value || cameraActive.value) return
  triggerFileSelect()
}

const appendFiles = (files) => {
  if (hasSelection.value) return
  const validImages = (files || []).filter((file) => file.type.startsWith('image/'))
  if (!validImages.length) {
    error.value = 'Please choose at least one image file.'
    return
  }
  results.value = null
  const newItems = validImages.map((file) => ({ file, preview: URL.createObjectURL(file) }))
  selectedFiles.value = [...selectedFiles.value, ...newItems]
  error.value = ''
}

const addFiles = (files) => {
  appendFiles(Array.from(files || []))
}

const onFileChange = (event) => {
  if (hasSelection.value) {
    event.target.value = ''
    return
  }
  addFiles(event.target.files)
  event.target.value = ''
}

const onDrop = (event) => {
  if (hasSelection.value) return
  addFiles(event.dataTransfer.files)
}

let cameraStream = null

const startCamera = async () => {
  if (hasSelection.value) return
  cameraError.value = ''
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraError.value = 'เบราว์เซอร์ไม่รองรับการเปิดกล้อง'
    return
  }
  try {
    if (cameraStream) stopCamera()
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    cameraActive.value = true
    await nextTick()
    if (videoRef.value) {
      videoRef.value.srcObject = cameraStream
      videoRef.value.muted = true
      videoRef.value.playsInline = true
      videoRef.value.onloadedmetadata = () => {
        videoRef.value?.play().catch(() => {})
      }
      await videoRef.value.play().catch(() => {})
    }
  } catch (err) {
    cameraError.value = err?.message || 'ไม่สามารถเปิดกล้องได้'
    cameraActive.value = false
    cameraStream = null
  }
}

const stopCamera = () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop())
    cameraStream = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  cameraActive.value = false
}

const capturePhoto = () => {
  cameraError.value = ''
  const videoEl = videoRef.value
  if (!videoEl || !cameraActive.value) {
    cameraError.value = 'ยังไม่มีกล้องพร้อมใช้งาน'
    return
  }

  const canvas = document.createElement('canvas')
  const width = videoEl.videoWidth || 1024
  const height = videoEl.videoHeight || 1024
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    cameraError.value = 'ไม่สามารถจับภาพได้'
    return
  }

  ctx.drawImage(videoEl, 0, 0, width, height)
  canvas.toBlob((blob) => {
    if (!blob) {
      cameraError.value = 'ไม่สามารถบันทึกภาพได้'
      return
    }
    const file = new File([blob], `capture-${Date.now()}.png`, { type: 'image/png' })
    appendFiles([file])
    stopCamera()
  }, 'image/png')
}

const processSelected = async () => {
  error.value = ''
  results.value = null

  if (!selectedFiles.value.length) {
    error.value = 'Please choose at least one image first.'
    return
  }

  const thresholdValue = Number(formInputs.value.threshold)
  const safeThreshold =
    formInputs.value.threshold === '' || formInputs.value.threshold === null || Number.isNaN(thresholdValue)
      ? 5
      : thresholdValue

  loading.value = true
  const useBatch = selectedFiles.value.length > 1
  const formData = new FormData()
  if (useBatch) {
    selectedFiles.value.forEach(({ file }) => formData.append('files', file))
  } else {
    formData.append('file', selectedFiles.value[0].file)
  }
  formData.append('formula', formInputs.value.formula || '')
  formData.append('lot_number', formInputs.value.lotNumber || '')
  formData.append('threshold', safeThreshold)

  try {
    const endpoint = useBatch ? 'batch-upload' : 'upload'
    const res = await fetch(`${apiUrl}/${endpoint}`, { method: 'POST', body: formData })
    const data = await res.json()
    if (!res.ok || !data.success) throw new Error(data.error || 'Processing failed')
    results.value = { ...data, mode: useBatch ? 'batch' : 'single' }
    activeIndex.value = 0
    clearSelectedFiles()
    await refreshHistory()
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
    checkHealth()
  }
}

const refreshHistory = async () => {
  try {
    const res = await fetch(`${apiUrl}/history`)
    const data = await res.json()
    history.value = data.items || []
  } catch (err) {
    // ignore silently
  }
}

const resetFilters = () => {
  filters.value = { lot: '', status: '' }
}

const acknowledge = () => {
  // No-op placeholder for UI parity
}

const scrollToUpload = () => {
  const el = document.getElementById('upload-card')
  if (el) el.scrollIntoView({ behavior: 'smooth' })
}

const nextItem = () => {
  if (!batchItems.value.length) return
  activeIndex.value = (activeIndex.value + 1) % batchItems.value.length
}

const prevItem = () => {
  if (!batchItems.value.length) return
  activeIndex.value = (activeIndex.value - 1 + batchItems.value.length) % batchItems.value.length
}

const nextSelected = () => {
  if (!selectedFiles.value.length) return
  selectedPreviewIndex.value = (selectedPreviewIndex.value + 1) % selectedFiles.value.length
}

const prevSelected = () => {
  if (!selectedFiles.value.length) return
  selectedPreviewIndex.value = (selectedPreviewIndex.value - 1 + selectedFiles.value.length) % selectedFiles.value.length
}

const checkHealth = async () => {
  backendStatus.value = 'checking'
  try {
    const res = await fetch(`${apiUrl}/health`)
    const data = await res.json()
    backendStatus.value = data.models_loaded ? 'ok' : 'warn'
  } catch {
    backendStatus.value = 'error'
  }
}

const openExportModal = () => {
  exportError.value = ''
  showExportModal.value = true
}

const closeExportModal = () => {
  if (exporting.value) return
  showExportModal.value = false
  exportError.value = ''
}

const toBackendDate = (isoDate) => {
  if (!isoDate) return ''
  const [year, month, day] = isoDate.split('-')
  return `${day}/${month}/${year}`
}

const downloadHistory = async () => {
  exportError.value = ''
  if (!exportRange.value.start || !exportRange.value.end) {
    exportError.value = 'โปรดเลือกช่วงวันที่ครบถ้วน'
    return
  }

  exporting.value = true
  try {
    const start = toBackendDate(exportRange.value.start)
    const end = toBackendDate(exportRange.value.end)
    const params = new URLSearchParams({ start, end })
    const res = await fetch(`${apiUrl}/history/export?${params.toString()}`)
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.error || 'ดาวน์โหลดไม่สำเร็จ')
    }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${start}_to_${end}.xlsx`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    closeExportModal()
  } catch (err) {
    exportError.value = err.message
  } finally {
    exporting.value = false
  }
}

onMounted(async () => {
  await Promise.all([checkHealth(), refreshHistory()])
})

onUnmounted(() => {
  stopCamera()
  clearSelectedFiles()
})
</script>
