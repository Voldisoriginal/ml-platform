<template>
  <div class="table-section">
    <DataTable 
      :value="runningModels" 
      :paginator="true" 
      :rows="10"
      paginatorTemplate="FirstPageLink PrevPageLink PageLinks NextPageLink LastPageLink CurrentPageReport RowsPerPageDropdown"
      :rowsPerPageOptions="[5,10,20]"
      responsiveLayout="scroll"
      class="p-datatable-striped"
      :loading="loading"
    >
      <Column field="model_id" header="Model ID" :sortable="true"></Column>
      <Column field="dataset_filename" header="Dataset" :sortable="true">
        <template #body="{data}">
          {{ formatFilename(data.dataset_filename) }}
        </template>
      </Column>
      <Column field="target_column" header="Target" :sortable="true"></Column>
      <Column field="model_type" header="Type" :sortable="true"></Column>
      
      <Column header="Metrics">
        <template #body="{data}">
          <div class="metrics">
            <span v-for="(value, key) in data.metrics" :key="key" class="metric-badge">
              {{ key }}: {{ value.toFixed(2) }}
            </span>
          </div>
        </template>
      </Column>

      <Column header="API">
        <template #body="{data}">
          <a :href="data.api_url" target="_blank" class="endpoint-link">
            <i class="pi pi-link"></i> Endpoint
          </a>
        </template>
      </Column>

      <Column header="Status">
        <template #body="{data}">
          <Tag 
            :value="data.status" 
            :severity="statusSeverity(data.status)" 
          />
        </template>
      </Column>

      <Column header="Actions">
        <template #body="{data}">
          <Button 
            icon="pi pi-stop" 
            class="p-button-danger"
            @click="stopInference(data.model_id)"
            v-tooltip="'Stop inference'"
          />
        </template>
      </Column>
    </DataTable>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import { useToast } from 'primevue/usetoast';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Button from 'primevue/button';
import Tag from 'primevue/tag';

const API_BASE_URL = 'http://localhost:8000';
const toast = useToast();
const loading = ref(false);

const props = defineProps({
  runningModels: {
    type: Array,
    required: true
  }
});

const emit = defineEmits(['model-stopped']);

const formatFilename = (filename) => {
  return filename?.split('_').slice(1).join('_') || '';
};

const statusSeverity = (status) => {
  return status === 'running' ? 'success' : 'danger';
};

const stopInference = async (modelId) => {
  try {
    loading.value = true;
    await axios.delete(`${API_BASE_URL}/stop_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Stopped',
      detail: 'Model inference container stopped',
      life: 3000
    });
    emit('model-stopped');
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: 'Failed to stop inference',
      life: 5000
    });
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.table-section {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  padding: 1.5rem;
}

.metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.metric-badge {
  background: #f0f0f0;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

.endpoint-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #2196F3;
  text-decoration: none;
}

.endpoint-link:hover {
  text-decoration: underline;
}
</style>