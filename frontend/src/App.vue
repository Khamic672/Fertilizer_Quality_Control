<template>
  <div class="app-shell">
    <header class="hero">
      <div class="hero-text">
        <p class="eyebrow">Local-only workflow</p>
        <h1>Fertilizer Quality Control</h1>
        <p class="lede">
          Upload a fertilizer image, segment the content, and estimate NPK values using your local checkpoints.
        </p>
        <div class="status-row">
          <span :class="['pill', backendStatusClass]">
            Backend: {{ backendLabel }}
          </span>
          <span class="pill muted">1024 x 1024 pipeline</span>
          <span class="pill muted">Vue + Flask</span>
        </div>
        <div class="pipeline">
          <div v-for="step in pipelineSteps" :key="step.title" class="pipeline-step">
            <div class="step-dot" />
            <div>
              <p class="step-title">{{ step.title }}</p>
              <p class="step-subtitle">{{ step.subtitle }}</p>
            </div>
          </div>
        </div>
      </div>
      <div class="hero-card">
        <h3>Latest output</h3>
        <p class="muted-text">Shows after a successful run.</p>
        <div v-if="results" class="mini-preview">
          <img :src="results.original" alt="Original preview" />
          <img :src="results.segmentation" alt="Segmentation preview" />
        </div>
        <div v-else class="mini-placeholder">
          <p>No run yet. Drop an image to start.</p>
        </div>
        <div class="meta-line">
          <span>Device</span>
          <strong>{{ backendDevice || 'unknown' }}</strong>
        </div>
        <div class="meta-line">
          <span>Models</span>
          <strong>{{ backendModelsLoaded ? 'loaded' : 'pending' }}</strong>
        </div>
      </div>
    </header>

    <main class="content">
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Step 1 - Upload</p>
            <h2>Accept image</h2>
            <p class="muted-text">JPEG or PNG. The backend will resize to 1024 x 1024.</p>
          </div>
        </div>
        <ImageUpload @file-selected="handleFileSelected" :loading="loading" />
      </section>

      <section v-if="results" class="panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">Step 2 - Output</p>
            <h2>Results</h2>
            <p class="muted-text">Includes original, mask segment, and NPK regression output.</p>
          </div>
          <button class="ghost-button" @click="resetResults">Clear</button>
        </div>
        <ResultDisplay :results="results" />
      </section>

      <section v-if="error" class="panel error">
        <div class="panel-head">
          <div>
          <p class="eyebrow">Issue</p>
            <h2>Something went wrong</h2>
          </div>
          <button class="ghost-button" @click="error = ''">Dismiss</button>
        </div>
        <p class="error-text">{{ error }}</p>
      </section>
    </main>

    <LoadingSpinner v-if="loading" />
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import ImageUpload from './components/ImageUpload.vue'
import ResultDisplay from './components/ResultDisplay.vue'
import LoadingSpinner from './components/LoadingSpinner.vue'

const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000/api'
const loading = ref(false)
const results = ref(null)
const error = ref('')
const backendStatus = ref('checking')
const backendDevice = ref('')
const backendModelsLoaded = ref(false)

const pipelineSteps = [
  { title: 'Accept image', subtitle: 'Upload JPEG or PNG' },
  { title: 'Process to 1024x1024', subtitle: 'Resize and normalize' },
  { title: 'Mask segment', subtitle: 'UNet initial predict' },
  { title: 'Regression', subtitle: 'NPK estimation' },
  { title: 'Postprocess output', subtitle: 'Encode results & metadata' },
  { title: 'Expose endpoints', subtitle: '/api/upload, /api/health' }
]

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

const resetResults = () => {
  results.value = null
}

const checkHealth = async () => {
  backendStatus.value = 'checking'
  try {
    const res = await fetch(`${apiUrl}/health`)
    const data = await res.json()
    backendStatus.value = 'ok'
    backendDevice.value = data.device
    backendModelsLoaded.value = Boolean(data.models_loaded)
  } catch (err) {
    backendStatus.value = 'error'
    error.value = 'Cannot reach backend. Make sure Flask server is running on port 5000.'
  }
}

const handleFileSelected = async (file) => {
  loading.value = true
  error.value = ''
  results.value = null

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch(`${apiUrl}/upload`, {
      method: 'POST',
      body: formData
    })

    const data = await response.json()
    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Processing failed')
    }
    results.value = data
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
    checkHealth()
  }
}

onMounted(() => {
  checkHealth()
})
</script>
