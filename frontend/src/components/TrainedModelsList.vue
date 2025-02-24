<template>
    <div>
      <h2>Trained Models</h2>
      <div v-if="models.length === 0">No trained models yet.</div>
      <div v-else class="model-cards">
        <div v-for="model in models" :key="model.id" class="model-card" @click="showModelDetails(model)">
          <div class="model-card-header">
            <!-- Название датасета (если есть) или ID -->
            <h3>{{ model.dataset_filename || model.id }}</h3>
            <p>Model Type: {{ model.model_type }}</p>
          </div>
          <div class="model-card-metrics">
              <!-- Краткий вывод метрик -->
              <p v-if="model.metrics && model.metrics.r2_score">R² Score: {{ model.metrics.r2_score.toFixed(4) }}</p>
              <p v-if="model.metrics && model.metrics.mse">MSE: {{ model.metrics.mse.toFixed(4) }}</p>
          </div>
        </div>
      </div>
  
      <!-- Модальное окно с деталями модели -->
      <div v-if="selectedModel" class="modal-overlay">
        <div class="modal">
          <div class="modal-header">
            <h2>Model Details</h2>
            <button @click="closeModal">Close</button>
          </div>
          <div class="modal-body">
            <p><strong>ID:</strong> {{ selectedModel.id }}</p>
            <p><strong>Dataset:</strong> {{ selectedModel.dataset_filename }}</p>
              <p>
                  <strong>Download Dataset:</strong>
                  <a :href="getDatasetDownloadLink(selectedModel.dataset_filename)" download>
                      Download
                  </a>
              </p>
            <p><strong>Target Column:</strong> {{ selectedModel.target_column }}</p>
            <p><strong>Model Type:</strong> {{ selectedModel.model_type }}</p>
  
            <!-- Параметры обучения -->
            <div v-if="selectedModel.train_settings">
              <p><strong>Train Settings:</strong></p>
              <ul>
                  <li>Train Size: {{ selectedModel.train_settings.train_size }}</li>
                  <li>Random State: {{ selectedModel.train_settings.random_state }}</li>
               </ul>
            </div>
  
              <!-- Параметры модели-->
            <div v-if="selectedModel.params && Object.keys(selectedModel.params).length > 0">
              <p><strong>Model Parameters:</strong></p>
               <ul>
                  <li v-for="(value, key) in selectedModel.params" :key="key">
                      {{ key }}: {{ value }}
                  </li>
              </ul>
            </div>
            <div  v-else> Parameters: No parameters</div>
  
              <!-- Метрики-->
            <div v-if="selectedModel.metrics">
              <p><strong>Metrics:</strong></p>
               <ul>
                   <li v-for="(value, key) in selectedModel.metrics" :key="key">
                      {{ key }}: {{ value.toFixed(4) }}
                   </li>
               </ul>
  
            </div>
            <button @click="selectForInference">Select for Inference</button>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios';
  
  const API_BASE_URL = 'http://localhost:8000';
  
  export default {
    props: {
      models: {
        type: Array,
        required: true,
      },
    },
    data() {
      return {
        selectedModel: null,
  
      };
    },
    methods: {
      showModelDetails(model) {
        this.selectedModel = model;
      },
      closeModal() {
        this.selectedModel = null;
      },
      selectForInference() {
        this.$emit('model-selected-for-inference', this.selectedModel.id);
        this.closeModal(); // Закрываем модальное окно после выбора
      },
       getDatasetDownloadLink(filename) {
            return `${API_BASE_URL}/download_dataset/${filename}`;
      },
    },
  };
  </script>
  
  <style scoped>
  .model-cards {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
  }
  
  .model-card {
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 5px;
    width: 200px; /* Или другая ширина */
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .model-card:hover {
    background-color: #f0f0f0;
  }
  .model-card-header{
  
  }
  .model-card-metrics{
  
  }
  /* Модальное окно */
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
  }
  
  .modal {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    width: 80%; /* Можно настроить */
    max-width: 600px; /* Можно настроить */
  }
  .modal-header{
      display:flex;
      justify-content: space-between;
  
  }
  .modal-body{
  
  }
  /* Стили для кнопок и заголовков (на ваше усмотрение) */
  </style>
  