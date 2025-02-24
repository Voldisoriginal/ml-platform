<template>
  <div v-if="modelId" class="inference-container">
    <h2>Model Inference: {{ modelId }}</h2>
    <div v-if="inferenceInfo">
      <p>Inference is running:</p>
      <ul>
          <li>
              <strong>Upload CSV:</strong>
              <a :href="inferenceInfo.upload_url" target="_blank">{{ inferenceInfo.upload_url }}</a>
            </li>
            <li>
              <strong>Send JSON:</strong>
              <a :href="inferenceInfo.json_url" target="_blank">{{ inferenceInfo.json_url }}</a>
          </li>

      </ul>

      <button @click="stopInference">Stop Inference</button>
    </div>
    <div v-else-if="loading">
        Loading inference information...
      </div>
    <div v-else>
        Inference container not running.
      </div>
        <button @click="closeInference">Close</button>  <!--Кнопка закрытия -->
  </div>
</template>

<script>
import axios from 'axios';
const API_BASE_URL = 'http://localhost:8000';

export default {
  props: {
    modelId: {
      type: String,
      required: true,
    },
    featureNames: { // Ожидаем featureNames
      type: Array,
      required: true,
    }
  },
   data() {
    return {
        inferenceInfo: null,
        loading: false,
    };
  },
  methods: {
        async fetchInferenceInfo() {
          this.loading = true;
            try {
                const response = await axios.post(`${API_BASE_URL}/start_inference/${this.modelId}`);
                this.inferenceInfo = response.data;
            } catch (error) {
                console.error("Error fetching inference info:", error);
                this.inferenceInfo = null; // Сбрасываем
                if (error.response && error.response.status === 404) {
                    alert("Inference container not found or has expired.");
                } else {
                    alert("An error occurred while fetching inference information.");
                }

            } finally {
              this.loading = false;
            }
        },
        async stopInference() {
            try {
                await axios.delete(`${API_BASE_URL}/stop_inference/${this.modelId}`);
                this.inferenceInfo = null; // Сбрасываем информацию
                alert("Inference container stopped.");

            } catch (error) {
                console.error("Error stopping inference:", error);
                 if (error.response && error.response.status === 404) {
                    alert("Inference container not found."); // Уже может не быть
                }
                else{
                    alert("Failed to stop inference container.");
                }
            }
        },
        closeInference() {
            this.$emit('close-inference'); //  Событие для закрытия
        }
    },
    watch: {
        modelId(newModelId) {
            // При изменении modelId (если выбрали другую модель), заново загружаем
            if (newModelId) {
                this.fetchInferenceInfo();
            } else {
              this.inferenceInfo = null;
            }
        }
    },
    created() {
        this.fetchInferenceInfo(); // Загружаем при создании компонента
    },

};
</script>

<style scoped>
/* Добавьте свои стили */
.inference-container {
  border: 1px solid #ddd;
  padding: 15px;
  margin-top: 20px;
}
</style>
