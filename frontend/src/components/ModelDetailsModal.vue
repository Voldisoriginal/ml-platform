<template>
  <div class="modal-overlay" @click.self="close">
    <div class="modal">
      <div class="modal-header">
        <h2>Model Details</h2>
        <button @click="close">Close</button>
      </div>
      <div class="modal-body">
        <p><strong>ID:</strong> {{ model.id }}</p>
        <p><strong>Dataset:</strong> {{ formatFilename(model.dataset_filename) }}</p>
        <p>
          <strong>Download Dataset:</strong>
          <a :href="getDatasetDownloadLink(model.dataset_filename)" download>
            Download
          </a>
        </p>
        <p><strong>Target Column:</strong> {{ model.target_column }}</p>
        <p><strong>Model Type:</strong> {{ model.model_type }}</p>

        <!-- Train Settings -->
        <div v-if="model.train_settings">
          <p><strong>Train Settings:</strong></p>
          <ul>
            <li>Train Size: {{ model.train_settings.train_size }}</li>
            <li>Random State: {{ model.train_settings.random_state }}</li>
          </ul>
        </div>

        <!-- Model Parameters -->
        <div v-if="model.params && Object.keys(model.params).length > 0">
          <p><strong>Model Parameters:</strong></p>
          <ul>
            <li v-for="(value, key) in model.params" :key="key">
              {{ key }}: {{ value }}
            </li>
          </ul>
        </div>
        <div v-else>Parameters: No parameters</div>

        <!-- Metrics -->
        <div v-if="model.metrics">
          <p><strong>Metrics:</strong></p>
          <ul>
            <li v-for="(value, key) in model.metrics" :key="key">
              {{ key }}: {{ value.toFixed(4) }}
            </li>
          </ul>
        </div>
         <button @click="selectForInference">Select for Inference</button>
      </div>
        </div>
    </div>
</template>

<script>

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default {
  props: {
    model: {
      type: Object,
      required: true,
    },
  },
  methods: {
    close() {
      this.$emit('close');
    },
      selectForInference() {
          this.$emit('close'); // close modal
          this.$parent.startInference(this.model.id); // Start inference at parent

      },
        getDatasetDownloadLink(filename) {
      return `${API_BASE_URL}/download_dataset/${filename}`;
    },
    formatFilename(filename) {
      if (!filename) {
        return '';
      }
      const parts = filename.split('_');
      if (parts.length > 1) {
        return parts.slice(1).join('_');
      }
      return filename;
    },
  },
};
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10; /* High z-index */
}

.modal {
  background-color: white;
  padding: 20px;
  border-radius: 5px;
    width: 80%;
  max-width: 600px;
}

.modal-header {
    display: flex;
    justify-content: space-between;
}
</style>
