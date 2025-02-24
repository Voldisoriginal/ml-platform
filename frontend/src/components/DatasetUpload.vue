<template>
  <div>
    <h2>Upload Dataset</h2>
    <input type="file" @change="handleFileUpload" accept=".csv" />
    <div v-if="errorMessage" class="error-message">{{ errorMessage }}</div>
  </div>
</template>

<script>
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default {
  name: 'DatasetUpload',
  data() {
    return {
      errorMessage: null,
    };
  },
  methods: {
    async handleFileUpload(event) {
      this.errorMessage = null;  // Сброс ошибки
      const file = event.target.files[0];
      if (!file) return;



      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'  // Важно для загрузки файлов!
          }
        });

        // Успешная загрузка
        this.$emit('dataset-uploaded', { filename: response.data.filename, columns: response.data.columns });

      } catch (error) {
         if (error.response) {
              this.errorMessage = error.response.data.detail; // Отображаем ошибку, полученную от API
          } else {
              this.errorMessage = "An error occurred during file upload."; // Общая ошибка
          }
          console.error("File upload failed:", error);
      }
    }
  }
};
</script>

<style scoped>
.error-message {
  color: red;
  margin-top: 10px;
}
</style>
