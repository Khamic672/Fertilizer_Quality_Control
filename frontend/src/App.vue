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
        <button class="ghost" @click="refreshHistory">History</button>
        <button class="primary" @click="scrollToUpload">Upload now</button>
      </div>
    </header>

    <main class="layout">
      <section id="upload-card" class="card upload-card">
        <div class="upload-left">
          <div class="drop-area" @click="triggerFileSelect" @dragover.prevent @drop.prevent="onDrop">
            <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileChange" />
            <div class="drop-inner">
              <div class="camera-circle">📷</div>
              <div>
                <h3>Upload a photo or take a new sample</h3>
                <p class="muted">We resize to 1024x1024 and run segmentation + regression locally.</p>
              </div>
            </div>
            <div class="upload-buttons">
              <button class="ghost">Camera</button>
              <button class="ghost">Album</button>
              <button class="ghost">Samples</button>
            </div>
          </div>

          <div class="sample-strip">
            <div v-for="sample in samples" :key="sample.id" class="sample-thumb">
              <img :src="sample.image" alt="sample" />
              <p>{{ sample.label }}</p>
            </div>
          </div>

          <div class="action-row">
            <div class="selectors">
              <select v-model="filters.region">
                <option value="">Region</option>
                <option>North</option>
                <option>Central</option>
                <option>South</option>
              </select>
              <select v-model="filters.district">
                <option value="">District</option>
                <option>A</option>
                <option>B</option>
              </select>
              <select v-model="filters.crop">
                <option value="">Crop</option>
                <option>Maize</option>
                <option>Rice</option>
                <option>Wheat</option>
              </select>
            </div>
            <button class="primary" :disabled="loading" @click="processLast">Upload &amp; Segment</button>
          </div>

          <div v-if="error" class="alert error">
            <p>{{ error }}</p>
          </div>
        </div>

        <div class="upload-right" v-if="results">
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

        <div v-else class="upload-right placeholder">
          <p class="muted">Results will appear here after processing.</p>
        </div>
      </section>

      <section class="card history-card">
        <div class="history-head">
          <h3>Lab entries</h3>
          <div class="history-actions">
            <select v-model="filters.lot">
              <option value="">Lot</option>
              <option>Lot A</option>
              <option>Lot B</option>
            </select>
            <select v-model="filters.status">
              <option value="">Status</option>
              <option value="ok">OK</option>
              <option value="warn">Warn</option>
              <option value="bad">Bad</option>
            </select>
            <button class="ghost" @click="resetFilters">Reset</button>
            <button class="primary small" @click="refreshHistory">Update</button>
          </div>
        </div>

        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Date</th>
                <th>Moisture</th>
                <th>pH</th>
                <th>N%</th>
                <th>P%</th>
                <th>K%</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in filteredHistory" :key="row.id">
                <td>{{ row.name }}</td>
                <td>{{ row.date }}</td>
                <td>{{ row.moisture }}</td>
                <td>{{ row.ph }}</td>
                <td>{{ row.n.toFixed(2) }}</td>
                <td>{{ row.p.toFixed(2) }}</td>
                <td>{{ row.k.toFixed(2) }}</td>
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
const samples = ref([])
const backendStatus = ref('checking')

const filters = ref({
  region: '',
  district: '',
  crop: '',
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
  loading.value = true
  error.value = ''
  results.value = null

  const formData = new FormData()
  formData.append('file', file)

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

const loadSamples = async () => {
  try {
    const res = await fetch(`${apiUrl}/samples`)
    const data = await res.json()
    samples.value = data.items || []
  } catch {
    samples.value = []
  }
}

const resetFilters = () => {
  filters.value = { region: '', district: '', crop: '', lot: '', status: '' }
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
  await Promise.all([checkHealth(), refreshHistory(), loadSamples()])
})
</script>
