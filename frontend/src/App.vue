<template>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">QC</div>
        <div>
          <p class="brand-eyebrow">KOSA QC Data Interface</p>
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
            <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileChange" />
            <div class="drop-inner">
              <div class="camera-icon">📷</div>
              <p class="muted center">ถ่ายรูป หรืออัปโหลดรูปการผสมปุ๋ย</p>
            </div>
            <div class="upload-buttons">
              <button class="ghost pill-button">📷 ถ่ายรูป</button>
              <button class="ghost pill-button">⬆️ อัปโหลดรูป</button>
            </div>
          </div>

          <div class="input-card">
            <p class="section-label">สูตรปุ๋ยที่ต้องการวิเคราะห์</p>
            <div class="input-grid">
              <div class="field">
                <label>สูตรปุ๋ย (NPK)</label>
                <input v-model="formInputs.formula" placeholder="เช่น 15-15-15" />
              </div>
              <div class="field">
                <label>Lot number*</label>
                <input v-model="formInputs.lotNumber" placeholder="ระบุ" />
              </div>
              <div class="field">
                <label>Threshold (% error)*</label>
                <input v-model.number="formInputs.threshold" type="number" min="0" max="100" placeholder="ระบุ" />
              </div>
            </div>
          </div>

          <div class="action-row">
            <button class="primary" :disabled="loading" @click="processLast">Upload &amp; Segment</button>
          </div>

          <div v-if="error" class="alert error">
            <p>{{ error }}</p>
          </div>
        </div>

        <div class="upload-results" v-if="results">
          <div class="preview-card">
            <img :src="results.segmentation || results.original" alt="preview" />
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
const lastFile = ref(null)
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

const metrics = computed(() => {
  if (!results.value) return []
  const npk = results.value.npk || { N: 0, P: 0, K: 0 }
  const total = 30
  return [
    { key: 'n', label: 'N', value: npk.N, percent: Math.min(100, (npk.N / total) * 100), color: '#1f8f45' },
    { key: 'p', label: 'P', value: npk.P, percent: Math.min(100, (npk.P / total) * 100), color: '#1f8f45' },
    { key: 'k', label: 'K', value: npk.K, percent: Math.min(100, (npk.K / total) * 100), color: '#c0392b' }
  ]
})

const statusList = computed(() => results.value?.status || [])
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

const onFileChange = (event) => {
  const [file] = event.target.files
  if (file) {
    lastFile.value = file
    processFile(file)
  }
  event.target.value = ''
}

const onDrop = (event) => {
  const [file] = event.dataTransfer.files
  if (file) {
    lastFile.value = file
    processFile(file)
  }
}

const processLast = () => {
  if (lastFile.value) {
    processFile(lastFile.value)
  } else {
    error.value = 'Please choose a file first.'
  }
}

const processFile = async (file) => {
  error.value = ''
  results.value = null

  const thresholdValue = Number(formInputs.value.threshold)
  if (!formInputs.value.lotNumber) {
    error.value = 'Please enter a lot number before uploading.'
    return
  }
  if (formInputs.value.threshold === '' || formInputs.value.threshold === null || Number.isNaN(thresholdValue)) {
    error.value = 'Please enter a valid threshold percentage.'
    return
  }

  loading.value = true
  const formData = new FormData()
  formData.append('file', file)
  formData.append('formula', formInputs.value.formula || '')
  formData.append('lot_number', formInputs.value.lotNumber)
  formData.append('threshold', thresholdValue)

  try {
    const res = await fetch(`${apiUrl}/upload`, { method: 'POST', body: formData })
    const data = await res.json()
    if (!res.ok || !data.success) throw new Error(data.error || 'Processing failed')
    results.value = data
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
