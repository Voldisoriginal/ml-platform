<template>
  <div class="models-page">
    <h1>Models</h1>

    <div class="controls-section">
      <Button 
        label="Train New Model" 
        icon="pi pi-plus" 
        @click="goToTraining"
      />
      <InputText 
        v-model="searchQuery" 
        placeholder="Search models..." 
        class="search-input"
      />
    </div>

    <!-- Trained Models Table -->
    <div class="table-section">
      <h2>Trained Models</h2>
      <DataTable 
        :value="filteredTrainedModels" 
        :paginator="true" 
        :rows="10"
        paginatorTemplate="FirstPageLink PrevPageLink PageLinks NextPageLink LastPageLink CurrentPageReport RowsPerPageDropdown"
        :rowsPerPageOptions="[5,10,20]"
        responsiveLayout="scroll"
        class="p-datatable-striped"
      >
        <Column field="id" header="ID" :sortable="true"></Column>
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
        <Column header="Actions">
          <template #body="{data}">
            <Button 
              icon="pi pi-chart-line" 
              class="p-button-info"
              @click="showModelDetails(data)"
              v-tooltip="'View details'"
            />
            <Button 
              icon="pi pi-play" 
              class="p-button-success"
              @click="startInference(data.id)"
              v-tooltip="'Start inference'"
            />
          </template>
        </Column>
      </DataTable>
    </div>

    <!-- Running Models Table -->
    <div class="table-section">
      <h2>Running Models</h2>
      <DataTable 
        :value="runningModels" 
        class="p-datatable-gridlines"
        :loading="loadingRunningModels"
      >
        <Column field="id" header="Model ID"></Column>
        <Column field="status" header="Status">
          <template #body="{data}">
            <Tag :value="data.status" 
              :severity="statusSeverity(data.status)" 
            />
          </template>
        </Column>
        <Column field="start_time" header="Start Time">
          <template #body="{data}">
            {{ formatDateTime(data.start_time) }}
          </template>
        </Column>
        <Column header="API Endpoints">
          <template #body="{data}">
            <div class="endpoints">
              <a :href="data.upload_url" target="_blank" class="endpoint-link">
                <i class="pi pi-cloud-upload"></i> Predict
              </a>
              <a :href="data.json_url" target="_blank" class="endpoint-link">
                <i class="pi pi-code"></i> Docs
              </a>
            </div>
          </template>
        </Column>
        <Column header="Actions">
          <template #body="{data}">
            <Button 
              icon="pi pi-stop" 
              class="p-button-danger"
              @click="stopInference(data.id)"
              v-tooltip="'Stop inference'"
            />
          </template>
        </Column>
      </DataTable>
    </div>

    <!-- Model Details Dialog -->
    <Dialog 
      v-model:visible="showDetailsDialog" 
      header="Model Details" 
      :modal="true"
      :style="{ width: '50vw' }"
    >
      <div v-if="selectedModel" class="model-details">
        <div class="detail-item">
          <label>Model Type:</label>
          <span>{{ selectedModel.model_type }}</span>
        </div>
        <div class="detail-item">
          <label>Dataset:</label>
          <span>{{ formatFilename(selectedModel.dataset_filename) }}</span>
        </div>
        <div class="detail-item">
          <label>Target Column:</label>
          <span>{{ selectedModel.target_column }}</span>
        </div>
        <div class="detail-item">
          <label>Training Date:</label>
          <span>{{ formatDateTime(selectedModel.start_time) }}</span>
        </div>
        <div class="detail-item">
          <label>Train Size:</label>
          <span>{{ selectedModel.train_settings.train_size }}</span>
        </div>
        <div class="detail-item">
          <label>Random State:</label>
          <span>{{ selectedModel.train_settings.random_state }}</span>
        </div>
        
        <div v-if="selectedModel.params && Object.keys(selectedModel.params).length > 0">
          <p><strong>Model Parameters:</strong></p>
          <ul>
            <li v-for="(value, key) in selectedModel.params" :key="key">
              {{ key }}: {{ value }}
            </li>
          </ul>
        </div>
        <div v-else>Parameters: No parameters</div>

        <h3>Metrics</h3>
        <div class="metrics-grid">
          <div 
            v-for="(value, key) in selectedModel.metrics" 
            :key="key"
            class="metric-card"
          >
            <div class="metric-title">{{ key }}</div>
            <div class="metric-value">{{ value.toFixed(4) }}</div>
          </div>
        </div>
      </div>
      <template #footer>
        <Button 
          label="Close" 
          icon="pi pi-times" 
          @click="showDetailsDialog = false" 
          class="p-button-text"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import axios from 'axios';
import { useRouter } from 'vue-router';
import { useToast } from 'primevue/usetoast';

// PrimeVue Components
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Button from 'primevue/button';
import Dialog from 'primevue/dialog';
import InputText from 'primevue/inputtext';
import Tag from 'primevue/tag';
import Tooltip from 'primevue/tooltip';

const API_BASE_URL = 'http://localhost:8000';
const router = useRouter();
const toast = useToast();

// Data
const searchQuery = ref('');
const trainedModels = ref([]);
const runningModels = ref([]);
const loadingRunningModels = ref(false);
const showDetailsDialog = ref(false);
const selectedModel = ref(null);

// Fetch data on mount
onMounted(async () => {
  await fetchTrainedModels();
  await fetchRunningModels();
});

// Computed
const filteredTrainedModels = computed(() => {
  const query = searchQuery.value.toLowerCase();
  return trainedModels.value.filter(model =>
    model.id.toLowerCase().includes(query) ||
    model.dataset_filename.toLowerCase().includes(query) ||
    model.target_column.toLowerCase().includes(query)
  );
});

// Methods
const formatFilename = (filename) => {
  return filename?.split('_').slice(1).join('_') || '';
};

const statusSeverity = (status) => {
  return status === 'running' ? 'success' : 'danger';
};

const formatDateTime = (dateString) => {
  return new Date(dateString).toLocaleString();
};

const fetchTrainedModels = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/trained_models/`);
    trainedModels.value = response.data;
  } catch (error) {
    showError('Failed to fetch trained models');
  }
};

const fetchRunningModels = async () => {
  try {
    loadingRunningModels.value = true;
    const response = await axios.get(`${API_BASE_URL}/running_models/`);
    runningModels.value = response.data;
  } catch (error) {
    showError('Failed to fetch running models');
  } finally {
    loadingRunningModels.value = false;
  }
};

const showModelDetails = (model) => {
  selectedModel.value = model;
  showDetailsDialog.value = true;
};

const startInference = async (modelId) => {
  try {
    await axios.post(`${API_BASE_URL}/start_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Started',
      detail: 'Model inference container started successfully',
      life: 3000
    });
    await fetchRunningModels();
  } catch (error) {
    showError('Failed to start inference');
  }
};

const stopInference = async (modelId) => {
  try {
    await axios.delete(`${API_BASE_URL}/stop_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Stopped',
      detail: 'Model inference container stopped',
      life: 3000
    });
    await fetchRunningModels();
  } catch (error) {
    showError('Failed to stop inference');
  }
};

const goToTraining = () => {
  router.push('/');
};

const showError = (message) => {
  toast.add({
    severity: 'error',
    summary: 'Error',
    detail: message,
    life: 5000
  });
};
</script>

<style scoped>
.models-page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.controls-section {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  align-items: center;
}

.search-input {
  width: 300px;
}

.table-section {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.table-section h2 {
  margin-top: 0;
  color: #2c3e50;
  border-bottom: 2px solid #eee;
  padding-bottom: 0.5rem;
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

.endpoints {
  display: flex;
  gap: 1rem;
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

.model-details {
  display: grid;
  gap: 1rem;
}

.detail-item {
  display: grid;
  grid-template-columns: 120px 1fr;
  align-items: center;
  padding: 0.5rem 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.metric-card {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 6px;
  text-align: center;
}

.metric-title {
  font-weight: 600;
  color: #495057;
  margin-bottom: 0.5rem;
}

.metric-value {
  font-size: 1.25rem;
  color: #2196F3;
}
</style>