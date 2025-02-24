<template>
  <div>
    <h2>Training Progress</h2>

    <div v-if="isTraining">
      <p>Training in progress...</p>
      <p>Task ID: {{ taskId }}</p>
      <p>Status: {{ taskStatus }}</p>
      <!-- Индикатор загрузки (spinner) -->
      <div v-if="error">
        <h3>Error Details:</h3>
        <p>Type: {{ error.exc_type }}</p>
        <p>Message: {{ error.exc_message }}</p>
        <!-- <pre v-if="error.exc_traceback">{{ error.exc_traceback }}</pre>  -->
      </div>
    </div>

    <div v-else-if="result">
      <h3>Training Results:</h3>
      <p>Model ID: {{ result.id }}</p>
      <p>Model Type: {{ result.model_type }}</p>
      <p>Metrics:</p>
      <ul>
        <li v-for="(value, key) in result.metrics" :key="key">
          {{ key }}: {{ value }}
        </li>
      </ul>
      <!-- Отображение графиков -->
    </div>
     <div v-else-if="error && !isTraining">
        <h3>Error Details:</h3>
        <p>Type: {{ error.exc_type }}</p>
        <p>Message: {{ error.exc_message }}</p>
        <!-- <pre v-if="error.exc_traceback">{{ error.exc_traceback }}</pre>  -->
      </div>
  </div>
</template>

<script>
import axios from 'axios';
const API_BASE_URL = 'http://localhost:8000';

export default {
  name: 'TrainingProgress',
  props: {
    isTraining: {
      type: Boolean,
      required: true,
    },
    result: {
      type: Object,
      required: false,
    },
    taskId: {
      type: String,
      required: false,
    },
  },
  data() {
    return {
      taskStatus: null,
      pollInterval: null,
      error: null, // Добавляем поле для хранения информации об ошибке
    };
  },
  watch: {
    taskId(newTaskId) {
      if (newTaskId) {
        this.startPolling();
      } else {
        this.stopPolling();
        this.taskStatus = null; // Сбрасываем статус
        this.error = null;     // Сбрасываем ошибку
      }
    },
  },
  methods: {
    async checkTaskStatus() {
      try {
        const response = await axios.get(`${API_BASE_URL}/train_status/${this.taskId}`);
        this.taskStatus = response.data.status;

        if (response.data.status === 'SUCCESS') {
          this.$emit('training-complete', response.data.result);
          this.stopPolling();
          this.$parent.isTraining = false;
          this.$parent.trainingResult = response.data.result;
          this.$parent.fetchTrainedModels();
          this.error = null; // Очищаем ошибку при успехе

        } else if (response.data.status === 'FAILURE') {
          this.error = response.data.error; // Сохраняем информацию об ошибке
          // this.stopPolling(); //Не останавливаем, а показываем ошибку + продолжаем опрос
          console.error("Training failed:", this.error);
          // alert('Training failed!');  //  Лучше не использовать alert

        } else {
            this.error = null; // Очищаем ошибку, если статус не FAILURE
        }

      } catch (error) {
        console.error("Error checking task status:", error);
        this.error = { exc_type: "NetworkError", exc_message: "Failed to fetch status." }; //  Или другая информация
        //this.stopPolling(); //  Не останавливаем опрос при ошибке сети, пробуем снова
      }
    },
    startPolling() {
      this.stopPolling();
      this.pollInterval = setInterval(this.checkTaskStatus, 2000);
    },
    stopPolling() {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
    },
  },
  beforeUnmount() {
    this.stopPolling();
  },
};
</script>
