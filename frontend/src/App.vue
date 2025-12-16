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
          <div class="drop-area" @click="triggerFileSelect" @dragover.prevent @drop.prevent="onDrop">
            <input ref="fileInput" type="file" accept="image/*" multiple hidden @change="onFileChange" />
            <div class="drop-inner">
              <div class="camera-icon">📷</div>
              <p class="muted center">ถ่ายรูป หรืออัปโหลดหลายรูปในล็อตเดียว</p>
            </div>
            <div class="upload-buttons">
              <button class="ghost pill-button">📷 ถ่ายรูป</button>
              <button class="ghost pill-button">⬆️ อัปโหลดรูป</button>
            </div>
          </div>

          <div class="input-card" v-if="selectedFiles.length">
            <p class="section-label">รูปที่เลือก ({{ selectedFiles.length }})</p>
            <ul class="muted">
              <li v-for="file in selectedFiles" :key="file.name + file.size">
                {{ file.name }} — {{ Math.round(file.size / 1024) }} KB
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

          <div class="action-row">
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

            <div class="batch-grid">
              <div class="preview-card" v-for="(item, index) in batchItems" :key="(item.filename || index) + index">
                <p class="section-label">{{ item.filename || `ภาพที่ ${index + 1}` }}</p>
                <img :src="item.segmentation || item.original" alt="preview" />

                <div class="status-bars">
                  <div class="bar-row" v-for="metric in buildMetrics(item.npk)" :key="metric.key">
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

                <div class="checklist mini" :class="statusTone(item.status_level)">
                  <p class="checklist-title">
                    <span>สถานะ</span>
                  </p>
                  <ul>
                    <li v-for="status in item.status" :key="status.message" :class="status.level">
                      <span class="dot" />
                      <span>{{ status.message }}</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </template>

          <template v-else>
            <div class="preview-card">
              <img :src="primaryResult?.segmentation || primaryResult?.original" alt="preview" />
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
          <p class="muted">Results will appear here after processing.</p>
        </div>
      </section>

      <section class="card history-card">
        <div class="history-head">
          <h3>ประวัติการวิเคราะห์</h3>
          <button class="ghost icon-button">
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

    <LoadingSpinner v-if="loading" />
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import LoadingSpinner from './components/LoadingSpinner.vue'

const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000/api'

const fileInput = ref(null)
const loading = ref(false)
const error = ref('')
const results = ref(null)
const selectedFiles = ref([])
const history = ref([])
const backendStatus = ref('checking')
const formInputs = ref({
  formula: '',
  lotNumber: '',
  threshold: 0.5
})

const filters = ref({
  lot: '',
  status: ''
})

const isBatchResult = computed(() => results.value?.mode === 'batch')
const batchItems = computed(() => (isBatchResult.value ? results.value?.items || [] : []))
const primaryResult = computed(() => {
  if (isBatchResult.value) {
    return batchItems.value[0] || null
  }
  return results.value
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

const buildMetrics = (npk) => {
  const values = npk || { N: 0, P: 0, K: 0 }
  const total = 30
  return [
    { key: 'n', label: 'N', value: values.N, percent: Math.min(100, (values.N / total) * 100), color: '#1f8f45' },
    { key: 'p', label: 'P', value: values.P, percent: Math.min(100, (values.P / total) * 100), color: '#1f8f45' },
    { key: 'k', label: 'K', value: values.K, percent: Math.min(100, (values.K / total) * 100), color: '#c0392b' }
  ]
}

const statusTone = (level) => {
  if (level === 'bad') return 'bad'
  if (level === 'warn') return 'warn'
  return 'good'
}

const metrics = computed(() => buildMetrics(primaryResult.value?.npk))

const statusList = computed(() => primaryResult.value?.status || [])
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

const addFiles = (files) => {
  const incoming = Array.from(files || []).filter((file) => file.type.startsWith('image/'))
  if (!incoming.length) {
    error.value = 'Please choose at least one image file.'
    return
  }
  results.value = null
  selectedFiles.value = [...selectedFiles.value, ...incoming]
  error.value = ''
}

const onFileChange = (event) => {
  addFiles(event.target.files)
  event.target.value = ''
}

const onDrop = (event) => {
  addFiles(event.dataTransfer.files)
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
      ? 0.5
      : thresholdValue

  loading.value = true
  const useBatch = selectedFiles.value.length > 1
  const formData = new FormData()
  if (useBatch) {
    selectedFiles.value.forEach((file) => formData.append('files', file))
  } else {
    formData.append('file', selectedFiles.value[0])
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
    selectedFiles.value = []
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

onMounted(async () => {
  await Promise.all([checkHealth(), refreshHistory()])
})
</script>
