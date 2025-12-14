<template>
  <div class="loading-overlay">
    <div class="loading-content">
      <div class="spinner"></div>
      <h3>Analyzing Fertilizer...</h3>
      <p>{{ statusMessage }}</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LoadingSpinner',
  data() {
    return {
      statusMessage: 'Preprocessing image...',
      messages: [
        'Preprocessing image...',
        'Running segmentation model...',
        'Extracting features...',
        'Predicting NPK values...'
      ],
      currentIndex: 0
    }
  },
  mounted() {
    this.interval = setInterval(() => {
      this.currentIndex = (this.currentIndex + 1) % this.messages.length
      this.statusMessage = this.messages[this.currentIndex]
    }, 1500)
  },
  beforeUnmount() {
    clearInterval(this.interval)
  }
}
</script>

<style scoped>
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.loading-content {
  background: white;
  border-radius: 12px;
  padding: 3rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.spinner {
  width: 60px;
  height: 60px;
  margin: 0 auto 1.5rem;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-content h3 {
  color: #2c3e50;
  margin-bottom: 0.5rem;
  font-size: 1.5rem;
}

.loading-content p {
  color: #7f8c8d;
  font-size: 1rem;
}
</style>