<template>
  <div class="upload-wrapper">
    <label
      class="dropzone"
      :class="{ 'dropzone--disabled': loading }"
      @dragover.prevent
      @drop.prevent="onDrop"
      @click.prevent="triggerFileSelect"
    >
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        @change="onFileChange"
        :disabled="loading"
        hidden
      />
      <div class="dropzone-content">
        <div class="icon-circle">UP</div>
        <div>
          <p class="drop-title">Drop a fertilizer image</p>
          <p class="muted-text">or click to browse. We accept JPG/PNG.</p>
        </div>
      </div>
      <p class="help-text">The backend will resize to 1024 x 1024 before segmentation.</p>
    </label>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['file-selected'])
const props = defineProps({
  loading: {
    type: Boolean,
    default: false
  }
})

const fileInput = ref(null)

const handleFile = (file) => {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file.')
    return
  }
  emit('file-selected', file)
}

const onFileChange = (event) => {
  const [file] = event.target.files
  handleFile(file)
  event.target.value = ''
}

const onDrop = (event) => {
  const [file] = event.dataTransfer.files
  handleFile(file)
}

const triggerFileSelect = () => {
  if (fileInput.value && !props.loading) {
    fileInput.value.click()
  }
}
</script>
