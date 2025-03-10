<template>
  <div>
    <div v-if="runningModels.length === 0">No models are currently running.</div>
    <table v-else>
      <thead>
        <tr>
          <th>Model ID</th>
          <th>Dataset</th>
          <th>Target Column</th>
          <th>Model Type</th>
          <th>Metrics</th>
           <th>API URL</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="model in runningModels" :key="model.model_id">
          <td>{{ model.model_id }}</td>
          <td>{{ formatFilename(model.dataset_filename) }}</td>
          <td>{{ model.target_column }}</td>
          <td>{{ model.model_type }}</td>
          <td>
            <ul>
              <li v-for="(value, key) in model.metrics" :key="key">
                {{ key }}: {{ value.toFixed(4) }}
              </li>
            </ul>
          </td>
           <td>
            <a :href="model.api_url" target="_blank">API</a>
          </td>

          <td>
            <button @click="stopInference(model.model_id)">Stop</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default {
  props: {
    runningModels: {
      type: Array,
      required: true
    }
  },
    methods: {
        formatFilename(filename) {
          if (!filename) {
            return '';
          }
            const parts = filename.split('_');
            if (parts.length > 1) {
                 return parts.slice(1).join('_');
            }
          return filename
        },

     async stopInference(modelId) {
      try {
        await axios.delete(`${API_BASE_URL}/stop_inference/${modelId}`);
        // Оповещаем родительский компонент об остановке, чтобы обновить список
        this.$emit('inference-stopped', modelId);  // Важно!
          // Обновляем локальный список (более оптимальный способ)
        this.$parent.fetchRunningModels(); // Вызов метода родителя
      } catch (error) {
        console.error('Error stopping inference:', error);
         if (error.response && error.response.status === 404) {
                    alert("Inference container not found."); // Уже может не быть
                }
                else{
                    alert("Failed to stop inference container.");
                }
      }
    },
  }
};
</script>

<style scoped>
/* Добавьте стили для таблицы, если необходимо */
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

th {
  background-color: #f2f2f2;
}
</style>
