<template>
  <div class="dataset-detail-view" v-if="!loadingDataset && dataset">
    <!-- Header Section -->
    <div class="header-section">
      <div class="header-left">
        <h1 class="dataset-title">{{ dataset.name }}</h1>
        <span class="creation-date">Created: {{ formattedDate(dataset.upload_date) }}</span>
      </div>
      <div class="header-right">
        <Button 
          label="Train" 
          icon="pi pi-download" 
          @click="goToTrain(dataset.id)" 
          class="p-button-outlined p-button-secondary" 
        />
        <Button 
          label="Download" 
          icon="pi pi-download" 
          @click="downloadDataset(dataset.filename)" 
          class="p-button-outlined p-button-secondary" 
        />
        <Button 
          icon="pi pi-ellipsis-v" 
          class="p-button-text p-button-secondary" 
          @click="toggleShareMenu" 
          aria-haspopup="true" 
          aria-controls="share_menu"
        />
        <Menu ref="shareMenu" id="share_menu" :model="shareMenuItems" :popup="true" />
      </div>
    </div>

    <!-- Description Section -->
    <div class="description-section">
      <div class="image-container">
        <Image 
          :src="getImageUrl(dataset)" 
          :alt="dataset.name" 
          preview 
          class="dataset-image-detail"
        />
      </div>
      <div class="description-content">
        <h2>Description</h2>
        <p v-if="dataset.description">{{ dataset.description }}</p>
        <p v-else><i>No description provided.</i></p>
      </div>
    </div>

    <!-- Tabs Section -->
    <TabView v-model:activeIndex="activeTabIndex">
      <!-- Data Card Tab -->
      <TabPanel header="Data Card">
        <div class="data-card-content">
          <div class="data-card-grid">
            <div class="info-item">
              <label>Author:</label>
              <span>{{ dataset.author || 'N/A' }}</span>
            </div>
            <div class="info-item">
              <label>Target Variable:</label>
              <Tag :value="dataset.target_variable || 'Not Set'" severity="info" />
            </div>
            <div class="info-item">
              <label>File Name:</label>
              <span>{{ dataset.filename }}</span>
            </div>
             <div class="info-item">
                <label>File Size:</label>
                <span>{{ dataset.file_size ? formatFileSize(dataset.file_size) : 'N/A' }}</span>
            </div>
             <div class="info-item">
                <label>Rows:</label>
                <span>{{ dataset.row_count !== undefined ? dataset.row_count : 'N/A' }}</span>
            </div>
            <div class="info-item">
                <label>Columns Count:</label>
                <span>{{ dataset.columns ? dataset.columns.length : 'N/A' }}</span>
            </div>
          </div>

          <h3>Columns</h3>
           <DataTable 
            v-if="columnInfo && columnInfo.length > 0"
            :value="columnInfo" 
            responsiveLayout="scroll" 
            class="p-datatable-sm"
            >
                <Column field="name" header="Name"></Column>
                <Column field="type" header="Type"></Column>
            </DataTable>
            <p v-else>Column type information not available.</p>


          <h3>Data Preview (First 20 Rows)</h3>
          <ProgressSpinner v-if="loadingPreview" style="width:50px;height:50px" strokeWidth="8" />
          <div v-else-if="previewError" class="error-message">
            Could not load data preview: {{ previewError }}
          </div>
           <DataTable 
            v-else-if="filePreview && filePreview.length > 0"
            :value="filePreview" 
            responsiveLayout="scroll" 
            scrollable 
            scrollHeight="400px" 
            class="p-datatable-sm preview-table-detail"
           >
            <Column v-for="col in dataset.columns" :key="col" :field="col" :header="col"></Column>
          </DataTable>
          <p v-else>No data preview available.</p>
        </div>
      </TabPanel>

      <!-- Training Results Tab -->
       <TabPanel header="Training Results">
         <ProgressSpinner v-if="loadingResults" style="width:50px;height:50px" strokeWidth="8" />
          <div v-else-if="trainingResultsError" class="error-message">
            Could not load training results: {{ trainingResultsError }}
          </div>
         <div v-else>
             <p v-if="!trainingResults || trainingResults.length === 0">
                 No training results found for this dataset yet.
             </p>
            <DataTable
                v-else
                :value="trainingResults"
                responsiveLayout="scroll"
                class="p-datatable-striped"
                >
                <Column field="model_type" header="Model Type" :sortable="true"></Column>
                 <Column field="target_column" header="Target" :sortable="true">
                     <template #body="{data}">
                         <Tag :value="data.target_column" />
                    </template>
                 </Column>
                <Column header="Metrics">
                    <template #body="{data}">
                        <div class="metrics-list">
                        <span v-for="(value, key) in data.metrics" :key="key" class="metric-chip">
                            {{ key }}: {{ value.toFixed ? value.toFixed(3) : value }}
                        </span>
                        </div>
                    </template>
                </Column>
                 <Column field="start_time" header="Trained On" :sortable="true">
                    <template #body="{data}">
                        {{ formatDateTime(data.start_time) }}
                    </template>
                </Column>
                 <Column header="Actions">
                    <template #body="{data}">
                        <Button
                        icon="pi pi-chart-line"
                        class="p-button-sm p-button-info p-button-text"
                        @click="showModelDetails(data)"
                        v-tooltip.top="'View Training Details'"
                        />
                        <Button
                        icon="pi pi-play"
                        class="p-button-sm p-button-success p-button-text"
                        @click="startInference(data.id)"
                        v-tooltip.top="'Start Inference'"
                         :disabled="isModelRunning(data.id)"
                        />
                    </template>
                </Column>
             </DataTable>
         </div>
        </TabPanel>
        
        <!-- Charts Tab (Placeholder) -->
        <TabPanel header="Charts">
            <p>Data visualizations will be displayed here (e.g., histograms, box plots for numerical columns).</p>
            <!-- TODO: Implement chart generation logic -->
        </TabPanel>
    </TabView>

    <!-- Model Details Dialog -->
    <Dialog
      v-model:visible="showDetailsDialog"
      header="Model Training Details"
      :modal="true"
      :style="{ width: '60vw' }"
    >
      <div v-if="selectedModel" class="model-details-dialog">
        <div class="detail-grid">
            <div class="detail-item"><label>Model ID:</label> <span>{{ selectedModel.id }}</span></div>
            <div class="detail-item"><label>Model Type:</label> <span>{{ selectedModel.model_type }}</span></div>
            <div class="detail-item"><label>Dataset:</label> <span>{{ formatFilename(selectedModel.dataset_filename) }}</span></div>
            <div class="detail-item"><label>Target Column:</label> <span>{{ selectedModel.target_column }}</span></div>
            <div class="detail-item"><label>Training Date:</label> <span>{{ formatDateTime(selectedModel.start_time) }}</span></div>
            <div class="detail-item"><label>Train Size:</label> <span>{{ selectedModel.train_settings?.train_size }}</span></div>
            <div class="detail-item"><label>Random State:</label> <span>{{ selectedModel.train_settings?.random_state }}</span></div>
        </div>

         <div v-if="selectedModel.params && Object.keys(selectedModel.params).length > 0" class="params-section">
          <h3>Model Parameters</h3>
          <pre>{{ JSON.stringify(selectedModel.params, null, 2) }}</pre>
        </div>
        <div v-else class="params-section">
            <h3>Model Parameters</h3>
            <p><i>No specific parameters recorded for this model type.</i></p>
        </div>

         <h3>Metrics</h3>
        <div class="metrics-grid-dialog">
          <div
            v-for="(value, key) in selectedModel.metrics"
            :key="key"
            class="metric-card-dialog"
          >
            <div class="metric-title-dialog">{{ key }}</div>
            <div class="metric-value-dialog">{{ value.toFixed ? value.toFixed(4) : value }}</div>
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
  <div v-else-if="loadingDataset" class="loading-container">
    <ProgressSpinner />
    <p>Loading dataset details...</p>
  </div>
   <div v-else-if="error" class="error-container">
       <Message severity="error" :closable="false">{{ error }}</Message>
       <Button label="Go Back to Datasets" icon="pi pi-arrow-left" @click="goBack" class="p-button-secondary" />
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import { useToast } from 'primevue/usetoast';

// PrimeVue Components
import Button from 'primevue/button';
import Menu from 'primevue/menu';
import Image from 'primevue/image';
import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';
import ProgressSpinner from 'primevue/progressspinner';
import Dialog from 'primevue/dialog';
import Message from 'primevue/message';
import Tooltip from 'primevue/tooltip';

const API_BASE_URL = 'http://localhost:8000';
const route = useRoute();
const router = useRouter();
const toast = useToast();

// State
const dataset = ref(null);
const trainingResults = ref([]);
const filePreview = ref([]);
const columnInfo = ref([]); // To store { name: 'col1', type: 'int64' }
const runningModels = ref([]); // To track which inference models are running
const loadingDataset = ref(true);
const loadingResults = ref(false);
const loadingPreview = ref(false);
const error = ref(null);
const trainingResultsError = ref(null);
const previewError = ref(null);
const activeTabIndex = ref(0);
const shareMenu = ref();
const showDetailsDialog = ref(false);
const selectedModel = ref(null);

const datasetId = computed(() => route.params.datasetId);

// Menu Items
const shareMenuItems = ref([
  {
    label: 'Share Link',
    icon: 'pi pi-share-alt',
    command: () => shareDataset()
  }
]);

const goToTrain = (datasetId)=>{
  router.push({path:'/', query:{dataset: datasetId}});
};
// --- Methods ---

const fetchDatasetDetails = async (id) => {
  loadingDataset.value = true;
  error.value = null;
  try {
    const response = await axios.get(`${API_BASE_URL}/dataset/${id}`);
    dataset.value = response.data;
     // Process column types if available from backend
     if (dataset.value.column_types) {
         columnInfo.value = Object.entries(dataset.value.column_types).map(([name, type]) => ({ name, type }));
    } else if (dataset.value.columns) {
        // Fallback if types aren't sent separately
         columnInfo.value = dataset.value.columns.map(name => ({ name, type: 'Unknown' }));
     }

    // Fetch related data once dataset details are loaded
    if (dataset.value?.filename) {
        fetchTrainingResults(dataset.value.filename);
        fetchDatasetPreview(dataset.value.filename);
        fetchRunningModels(); // Check running models status
    }

  } catch (err) {
    console.error('Error fetching dataset details:', err);
    error.value = `Failed to load dataset details: ${err.response?.data?.detail || err.message}`;
     dataset.value = null; // Clear dataset on error
  } finally {
    loadingDataset.value = false;
  }
};

const fetchTrainingResults = async (filename) => {
  loadingResults.value = true;
  trainingResultsError.value = null;
  try {
    const response = await axios.get(`${API_BASE_URL}/trained_models/search_sort?dataset_filename=${filename}&sort_by=start_time&sort_order=desc`);
    trainingResults.value = response.data;
  } catch (err) {
    console.error('Error fetching training results:', err);
    trainingResultsError.value = `Failed to load training results: ${err.response?.data?.detail || err.message}`;
     trainingResults.value = [];
  } finally {
    loadingResults.value = false;
  }
};

const fetchDatasetPreview = async (filename) => {
  loadingPreview.value = true;
  previewError.value = null;
  try {
    // **Assumption:** Backend provides a preview endpoint
    // If not, you'd need to implement download + client-side parse here
    const response = await axios.get(`${API_BASE_URL}/dataset_preview/${filename}?rows=20`);
    filePreview.value = response.data.preview;
     // Optionally update columnInfo if the preview endpoint provides types
     if(response.data.column_types) {
         columnInfo.value = Object.entries(response.data.column_types).map(([name, type]) => ({ name, type }));
     }
  } catch (err) {
    console.error('Error fetching data preview:', err);
     // Check if it's a 404, meaning the endpoint might not exist yet
     if (err.response?.status === 404) {
         previewError.value = 'Data preview feature not available yet.';
     } else {
        previewError.value = `Failed to load data preview: ${err.response?.data?.detail || err.message}`;
     }
    filePreview.value = [];
  } finally {
    loadingPreview.value = false;
  }
};

const fetchRunningModels = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/running_models/`);
        runningModels.value = response.data;
    } catch (error) {
        console.error('Failed to fetch running models status:', error);
        // Don't necessarily block the UI for this, maybe just log it
    }
};

const downloadDataset = async (filename) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/download_dataset/${filename}`, {
      responseType: 'blob',
    });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    toast.add({ severity: 'success', summary: 'Success', detail: 'Dataset download started.', life: 3000 });
  } catch (err) {
    console.error('Error downloading dataset:', err);
    toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to download dataset.', life: 3000 });
  }
};

const shareDataset = () => {
  const url = window.location.href;
  navigator.clipboard.writeText(url).then(() => {
    toast.add({ severity: 'success', summary: 'Success', detail: 'Link copied to clipboard!', life: 3000 });
  }, (err) => {
    console.error('Could not copy text: ', err);
    toast.add({ severity: 'warn', summary: 'Warning', detail: 'Could not copy link automatically.', life: 3000 });
  });
};

const toggleShareMenu = (event) => {
  shareMenu.value.toggle(event);
};

const startInference = async (modelId) => {
  try {
    await axios.post(`${API_BASE_URL}/start_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Started',
      detail: `Model ${modelId} inference started.`,
      life: 3000
    });
    // Refresh running models status
    await fetchRunningModels();
  } catch (error) {
    console.error('Error starting inference:', error);
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: `Failed to start inference: ${error.response?.data?.detail || error.message}`,
      life: 5000
    });
  }
};

const showModelDetails = (model) => {
  selectedModel.value = model;
  showDetailsDialog.value = true;
};

const goBack = () => {
    router.push({ name: 'Datasets' });
}

const isModelRunning = (modelId) => {
    return runningModels.value.some(m => m.model_id === modelId && m.status === 'running');
};


// --- Computed & Watchers ---

const getImageUrl = (ds) => {
  // Use the imageUrl provided by the backend API (which should point to /dataset/{id}/image)
  // Add the base URL only if it's a relative path
    if (ds?.imageUrl) {
        // Check if it's already an absolute URL
        if (ds.imageUrl.startsWith('http://') || ds.imageUrl.startsWith('https://')) {
            return ds.imageUrl;
        }
         // Assume it's a relative path from the API base
        return `${API_BASE_URL}${ds.imageUrl}`;
    }
    // Fallback placeholder
    return `${API_BASE_URL}/placeholder.png`;
};

const formattedDate = (dateString) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric', month: 'long', day: 'numeric'
  });
};

const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
};

const formatFileSize = (bytes) => {
  if (bytes === 0 || !bytes) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatFilename = (filename) => {
    // Basic formatting, remove potential UUID prefix
    return filename?.split('_').slice(1).join('_') || filename || '';
};

// Fetch data when the component mounts or the ID changes
onMounted(() => {
    if (datasetId.value) {
        fetchDatasetDetails(datasetId.value);
    } else {
        error.value = "No Dataset ID provided.";
        loadingDataset.value = false;
    }
});

// Watch for route changes if the user navigates between detail pages directly
watch(datasetId, (newId, oldId) => {
  if (newId && newId !== oldId) {
    fetchDatasetDetails(newId);
  }
});

</script>

<style scoped>
.dataset-detail-view {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

/* --- Header --- */
.header-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #dee2e6;
}

.header-left .dataset-title {
  margin: 0 0 0.25rem 0;
  color: #343a40;
}

.header-left .creation-date {
  font-size: 0.9rem;
  color: #6c757d;
}

.header-right {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

/* --- Description --- */
.description-section {
  display: flex;
  gap: 2rem;
  margin-bottom: 2rem;
  background-color: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  align-items: flex-start; /* Выравниваем по верху */
  /* Добавим overflow: hidden на всякий случай, чтобы обрезать вылезающее */
  /* Можно закомментировать, если не понадобится */
  /* overflow: hidden; */
}

.image-container {
  /* Задаем МАКСИМАЛЬНУЮ ширину */
  max-width: 300px; /* Например, 300px или 40% */
  width: 100%; /* Позволяем ему быть меньше, если нужно, но не больше max-width */
  flex-shrink: 0; /* Не позволяем контейнеру сжиматься */
  /* border: 1px dashed red; /* Временная рамка для отладки */
}

/* Используем :deep() для стилизации img ВНУТРИ компонента p-image */
.image-container :deep(img.p-image-preview), /* Стандартный класс PrimeVue для img в p-image */
.image-container :deep(img) { /* Общий селектор на всякий случай */
  display: block;       /* Убирает лишние отступы */
  max-width: 100%;      /* КЛЮЧ: Картинка не шире своего контейнера! */
  height: auto;         /* КЛЮЧ: Сохраняем пропорции */
  object-fit: contain;  /* КЛЮЧ: Вписывает картинку целиком, масштабируя ВНИЗ */
  max-height: 400px;    /* Опционально: Ограничить максимальную высоту */
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.description-content {
  flex-grow: 1; /* Занимает оставшееся место */
  min-width: 0; /* Позволяет текстовому блоку сжиматься */
  /* border: 1px dashed blue; /* Временная рамка для отладки */
}

.description-content h2 {
  margin-top: 0;
  margin-bottom: 1rem;
  color: #495057;
}

.description-content p {
  line-height: 1.6;
  color: #495057;
  word-break: break-word; /* Для переноса длинных слов, если нужно */
}

/* --- Tabs --- */
:deep(.p-tabview-nav) {
  background: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
}

:deep(.p-tabview .p-tabview-panels) {
  padding: 1.5rem 0.5rem; /* Add some padding inside tabs */
}

/* --- Data Card Tab --- */
.data-card-content {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}
.data-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem 1.5rem; /* row-gap column-gap */
  margin-bottom: 1.5rem;
}

.info-item {
    background-color: #ffffff;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    border: 1px solid #e9ecef;
}

.info-item label {
  display: block;
  font-weight: 600;
  color: #495057;
  margin-bottom: 0.3rem;
  font-size: 0.9rem;
}

.info-item span, .info-item .p-tag {
  font-size: 0.95rem;
  color: #6c757d;
}

.info-item .p-tag {
    vertical-align: middle;
}

.data-card-content h3 {
    margin-bottom: 0.75rem;
    margin-top: 1rem;
    color: #495057;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
}

.preview-table-detail :deep(.p-datatable-tbody > tr > td) {
  padding: 0.4rem 0.8rem !important; /* Smaller padding for preview */
  font-size: 0.85rem;
}

/* --- Training Results Tab --- */
.metrics-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.metric-chip {
  background-color: #e9ecef;
  color: #495057;
  padding: 0.25rem 0.6rem;
  border-radius: 12px;
  font-size: 0.8rem;
  white-space: nowrap;
}

/* --- Model Details Dialog --- */
.model-details-dialog {
  font-size: 0.95rem;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0.75rem 1.5rem;
    margin-bottom: 1.5rem;
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 6px;
}

.detail-item label {
  font-weight: 600;
  color: #343a40;
  margin-right: 0.5rem;
}

.params-section {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
.params-section h3 {
    margin-bottom: 0.5rem;
}
.params-section pre {
    background-color: #e9ecef;
    padding: 1rem;
    border-radius: 4px;
    max-height: 200px;
    overflow-y: auto;
    font-size: 0.85rem;
}

.metrics-grid-dialog {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.metric-card-dialog {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 6px;
  text-align: center;
  border: 1px solid #dee2e6;
}

.metric-title-dialog {
  font-weight: 600;
  color: #495057;
  margin-bottom: 0.5rem;
}

.metric-value-dialog {
  font-size: 1.2rem;
  color: #007bff;
}


/* --- Loading & Error --- */
.loading-container, .error-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 300px;
  text-align: center;
}
.loading-container p {
    margin-top: 1rem;
    font-size: 1.1rem;
    color: #6c757d;
}
.error-container .p-message {
    margin-bottom: 1.5rem;
}

.error-message {
    color: var(--red-500);
    background-color: var(--red-100);
    padding: 1rem;
    border-radius: 4px;
    border: 1px solid var(--red-200);
}

</style>