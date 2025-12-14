<template>
  <div class="results-grid">
    <div class="image-card">
      <p class="eyebrow">Main file</p>
      <h3>Original</h3>
      <img :src="results.original" alt="Original" />
    </div>
    <div class="image-card">
      <p class="eyebrow">Mask segment</p>
      <h3>Segmentation overlay</h3>
      <img :src="results.segmentation" alt="Segmentation" />
    </div>
    <div class="metrics-card">
      <p class="eyebrow">Regression</p>
      <h3>NPK estimates</h3>
      <div class="metric-row">
        <span>N</span>
        <strong>{{ results.npk.N.toFixed(2) }}</strong>
      </div>
      <div class="metric-row">
        <span>P</span>
        <strong>{{ results.npk.P.toFixed(2) }}</strong>
      </div>
      <div class="metric-row">
        <span>K</span>
        <strong>{{ results.npk.K.toFixed(2) }}</strong>
      </div>
      <div class="divider"></div>
      <p class="eyebrow">Postprocess</p>
      <div class="meta-row">
        <span>Classes detected</span>
        <strong>{{ results.metadata.classes_detected }}</strong>
      </div>
      <div class="meta-row">
        <span>Pixels analyzed</span>
        <strong>{{ results.metadata.pixels_analyzed }}</strong>
      </div>
      <div class="meta-row">
        <span>Size</span>
        <strong>{{ results.metadata.image_size }}</strong>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  results: {
    type: Object,
    required: true
  }
})
</script>
