<template>
  <div class="models-page">
    <h1>Running Models</h1>

    <div class="controls-section">
      <InputText 
        v-model="searchQuery" 
        placeholder="Search running models..." 
        class="search-input"
      />
    </div>

    <RunningModelsList 
      :runningModels="filteredRunningModels" 
      @model-stopped="fetchRunningModels"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import axios from 'axios';
import { useToast } from 'primevue/usetoast';
import InputText from 'primevue/inputtext';
import RunningModelsList from '@/components/RunningModelsList.vue';

const API_BASE_URL = 'http://localhost:8000';
const toast = useToast();

const searchQuery = ref('');
const runningModels = ref([]);

const filteredRunningModels = computed(() => {
  const query = searchQuery.value.toLowerCase();
  return runningModels.value.filter(model =>
    model.model_id.toLowerCase().includes(query) ||
    model.dataset_filename.toLowerCase().includes(query) ||
    model.target_column.toLowerCase().includes(query)
  );
});

const fetchRunningModels = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/running_models/`);
    runningModels.value = response.data;
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: 'Failed to fetch running models',
      life: 5000
    });
  }
};

onMounted(async () => {
  await fetchRunningModels();
});
</script>

<style scoped>
.models-page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.controls-section {
  margin-bottom: 2rem;
}

.search-input {
  width: 300px;
}
</style>