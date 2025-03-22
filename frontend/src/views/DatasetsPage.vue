<template>
  <div class="datasets-page">
    <h1>Datasets</h1>

    <div class="add-dataset-section">
      <Button label="Add Dataset" icon="pi pi-plus" @click="showAddDatasetDialog = true" />
    </div>
    <InputText v-model="searchQuery" placeholder="Search datasets..." class="search-input" />

    <Dialog v-model:visible="showAddDatasetDialog" modal header="Add Dataset" :style="{ width: '70vw' }">
    <div class="p-fluid">
      <!-- Шаг 1: Выбор файла -->
      <div v-if="currentStep === 1">
        <div class="p-field">
          <label for="dataset-file">Dataset File (CSV)</label>
          <FileUpload
            ref="fileUpload"
            mode="basic"
            :customUpload="true"
            @uploader="handleFileUpload"
            accept=".csv"
            :auto="false"
            chooseLabel="Select CSV"
          />
        </div>

        <div v-if="fileInfo" class="file-info-card">
          <div class="file-stats">
            <div class="stat-item">
              <span class="stat-label">File Name:</span>
              <span class="stat-value">{{ fileInfo.name }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">File Size:</span>
              <span class="stat-value">{{ fileInfo.size }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Total Rows:</span>
              <span class="stat-value">{{ fileInfo.rows }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Columns:</span>
              <span class="stat-value">{{ fileInfo.columns.join(', ') }}</span>
            </div>
          </div>

          <div class="preview-table">
            <h4>Preview (first 10 rows)</h4>
            <DataTable :value="previewData" class="p-datatable-sm" scrollable scrollHeight="200px">
              <Column v-for="col in fileInfo.columns" :key="col" :field="col" :header="col"></Column>
            </DataTable>
          </div>
        </div>
      </div>

        <!-- Шаг 2: Детали датасета -->
        <div v-if="currentStep === 2">
          <div class="p-field">
            <label for="dataset-name">Name</label>
            <InputText id="dataset-name" v-model="newDataset.name" required />
          </div>
          <div class="p-field">
            <label for="dataset-description">Description</label>
            <Textarea id="dataset-description" v-model="newDataset.description" rows="3" />
          </div>
          <div class="p-field">
            <label for="dataset-author">Author</label>
            <InputText id="dataset-author" v-model="newDataset.author" />
          </div>
          <div class="p-field">
    <label for="dataset-target">Target Variable</label>
    <Dropdown 
      id="dataset-target" 
      v-model="newDataset.targetVariable" 
      :options="fileInfo.columns" 
      placeholder="Select target variable"
      class="w-full"
      @change="onTargetChange"
    />
          </div>
          <div class="p-field">
            <label for="dataset-image">Image (optional)</label>
            <FileUpload
              ref="imageUpload"
              mode="basic"
              :customUpload="true"
              @uploader="handleImageUpload"
              accept="image/*"
              :auto="false"
              chooseLabel="Select Image"
            />
            <img v-if="newDataset.imageUrl" :src="newDataset.imageUrl" alt="Dataset Image Preview" class="image-preview" />
          </div>
        </div>
      </div>
      <template #footer>
        <Button 
          v-if="currentStep === 1" 
          label="Cancel" 
          icon="pi pi-times" 
          @click="cancelAddDataset" 
          class="p-button-text" 
        />
        <Button 
          v-if="currentStep === 2" 
          label="Back" 
          icon="pi pi-arrow-left" 
          @click="currentStep = 1" 
          class="p-button-text"
        />
        <Button 
          v-if="currentStep === 1" 
          label="Next" 
          icon="pi pi-arrow-right" 
          @click="currentStep = 2" 
          :disabled="!newDataset.file"
        />
        <Button 
          v-if="currentStep === 2" 
          label="Add" 
          icon="pi pi-check" 
          @click="addDataset" 
          :disabled="!isFormValid" 
          autofocus 
        />
      </template>
    </Dialog>

    <div class="dataset-cards">
        <Card v-for="dataset in filteredDatasets" :key="dataset.id" class="dataset-card" @click="showDatasetDetails(dataset.id)">
          <template #header>
            <img v-if="dataset.imageUrl" :src="API_BASE_URL + dataset.imageUrl" :alt="dataset.name" class="dataset-image"/>
            <img v-else :src="API_BASE_URL + '/placeholder.png'" alt="Placeholder" class="dataset-image"/>
          </template>
           <template #title>{{ dataset.name }}</template>
           <template #content>
            <p class="dataset-description" v-if="dataset.description">{{ truncatedDescription(dataset.description) }}</p>
            <p v-else class="no-description">No description provided</p>
          </template>
           </Card>
         </div>

    <Dialog v-model:visible="showDetailsDialog" modal :header="selectedDataset.name" :style="{ width: '70vw' }">
          <div class="dataset-details">
            <img v-if="selectedDataset.imageUrl" :src="API_BASE_URL + selectedDataset.imageUrl" :alt="selectedDataset.name" class="dataset-details-image"/>
            <img v-else :src="API_BASE_URL + '/placeholder.png'" alt="Placeholder Image" class="dataset-details-image"/>
            <div class="dataset-info">
                <p><strong>Name:</strong> {{ selectedDataset.name }}</p>
                <p><strong>Description:</strong> {{ selectedDataset.description }}</p>
                <p><strong>Author:</strong> {{ selectedDataset.author }}</p>
                <p><strong>Target Variable:</strong> {{ selectedDataset.target_variable }}</p>
                <p><strong>Upload Date:</strong> {{ formatDate(selectedDataset.upload_date) }}</p>
                <p><strong>Columns:</strong></p>
                <ul>
                  <li v-for="column in selectedDataset.columns" :key="column">{{ column }}</li>
                </ul>
                <div class="action-buttons">
                  <Button label="Download Dataset" icon="pi pi-download" @click="downloadDataset(selectedDataset.filename)" />
                  <Button label="Train Model" icon="pi pi-cog" @click="goToTraining(selectedDataset.id)" />
                </div>
            </div>
          </div>
          <div class="training-results" v-if="selectedDatasetTrainingResults.length > 0">
              <h3>Recent Training Results:</h3>
                <DataTable :value="selectedDatasetTrainingResults" responsiveLayout="scroll">
                  <Column field="model_type" header="Model Type"></Column>
                  <Column field="target_column" header="Target Column"></Column>
             <Column header="Metrics">
            <template #body="slotProps">
              <ul>
                <li v-for="(value, key) in slotProps.data.metrics" :key="key">
                  {{ key }}: {{ value }}
                </li>
              </ul>
            </template>
          </Column>
        </DataTable>
      </div>
      <template #footer>
          <Button label="Close" icon="pi pi-times" @click="showDetailsDialog = false" class="p-button-text" />
      </template>
  </Dialog>

  </div>
</template>

<script setup>
import Dropdown from 'primevue/dropdown';
import { ref, computed, onMounted, reactive } from 'vue';
import axios from 'axios';
import Card from 'primevue/card';
import Dialog from 'primevue/dialog';
import InputText from 'primevue/inputtext';
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import { useToast } from "primevue/usetoast";
import FileUpload from 'primevue/fileupload';
import { parse } from 'papaparse';
import { useRouter } from 'vue-router';


const toast = useToast();
const datasets = ref([]);
const showAddDatasetDialog = ref(false);
const showDetailsDialog = ref(false);
const selectedDataset = ref({});
const selectedDatasetTrainingResults = ref([]);
const searchQuery = ref('');
const router = useRouter();

const currentStep = ref(1);
const csvColumns = ref([]);

const fileUpload = ref(null);
const imageUpload = ref(null);

const newDataset = reactive({
  name: '',
  description: '',
  author: '',
  target_variable: '',
  file: null,
  imageUrl: null,
  imageFile: null,
});
const API_BASE_URL = 'http://localhost:8000';
const isFormValid = computed(() => {
  return newDataset.name && newDataset.file && newDataset.target_variable;
  });

const previewContent = ref(null);

const fetchDatasets = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/datasets/`);
    console.log("Response from /datasets/:", response);

    if (response.headers['content-type']?.includes('application/json')) {
        if (Array.isArray(response.data)) {
            datasets.value = response.data.map(dataset => ({
                ...dataset,
                imageUrl: dataset.imageUrl || null, // Use imageUrl sent from the backend
            }));
         } else {
            console.error("Unexpected response format (not an array) from /datasets/:", response.data);
            toast.add({ severity: 'error', summary: 'Error', detail: 'Received unexpected data format (not an array).', life: 3000 });
            datasets.value = [];
        }

    } else {
        console.error("Received non-JSON response from /datasets/:", response);
        toast.add({ severity: 'error', summary: 'Error', detail: 'Received non-JSON response from server.', life: 3000 });
        datasets.value = [];
    }
    // Remove fetchDatasetImages call.  No longer needed.

  } catch (error) {
    console.error('Error fetching datasets:', error);
    toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to fetch datasets', life: 3000 });
  }
};


const onTargetChange = (event) => {
  console.log('Selected value:', event.value);
  newDataset.target_variable = event.value;
};

const filteredDatasets = computed(() => {
  if (!searchQuery.value) {
    return datasets.value;
  }
  const query = searchQuery.value.toLowerCase();
  return datasets.value.filter(dataset =>
    dataset.name.toLowerCase().includes(query) ||
    (dataset.description && dataset.description.toLowerCase().includes(query)) ||
    (dataset.author && dataset.author.toLowerCase().includes(query))
  );
});


const handleImageUpload =  (event) => {
  const file = event.files[0];
    if (file){
      newDataset.imageFile = file;
      newDataset.imageUrl = URL.createObjectURL(file); // Create preview URL
    } else {
      newDataset.imageFile = null;
      newDataset.imageUrl = null;
    }
};

const addDataset = async () => {
  console.log('Sending target variable:', newDataset.targetVariable);
  if (!newDataset.file) {
      toast.add({ severity: 'warn', summary: 'Warning', detail: 'Please select a dataset file.', life: 3000 });
    return;
  }

  const formData = new FormData();
  formData.append('file', newDataset.file);
  formData.append('name', newDataset.name);
  
  if (newDataset.description) formData.append('description', newDataset.description);
  if (newDataset.author) formData.append('author', newDataset.author);
  if (newDataset.target_variable) formData.append('target_variable', newDataset.target_variable);
  if(newDataset.imageFile) formData.append('image', newDataset.imageFile);
  formData.append('target_variable', newDataset.target_variable);

  try {
    const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    toast.add({severity: 'success', summary: 'Success', detail: 'Dataset added successfully!', life: 3000});
    resetForm();
    showAddDatasetDialog.value = false;
    fetchDatasets();

  } catch (error) {
    //console.error('Error adding dataset:', error);
     //const detail = error.response?.data?.detail || 'Failed to add dataset';
     //toast.add({severity: 'error', summary: 'Error', detail: detail, life: 5000});
  }
};

const cancelAddDataset = () => {
    resetForm();
    showAddDatasetDialog.value = false;
};

const resetForm = () => {
    currentStep.value = 1;
    csvColumns.value = [];
    if (fileUpload.value) fileUpload.value.clear();
  if (imageUpload.value) imageUpload.value.clear();
  
  // Сбрасываем остальные значения
  currentStep.value = 1;
  newDataset.name = '';
  newDataset.description = '';
  newDataset.author = '';
  newDataset.targetVariable = '';
  newDataset.file = null;
  newDataset.imageUrl = null;
  newDataset.imageFile = null;
  fileInfo.value = null;
  previewData.value = [];
  csvColumns.value = [];
};



const showDatasetDetails = async (datasetId) => {
    try{
      const response = await axios.get(`${API_BASE_URL}/dataset/${datasetId}`);
      selectedDataset.value = response.data;
      showDetailsDialog.value = true;
        fetchTrainingResults(datasetId);
    } catch (error) {
      console.error('Error fetching dataset details:', error);
        toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to fetch dataset details', life: 3000 });
    }
 };

const fetchTrainingResults = async (datasetId) => {
    try {
      const dataset = datasets.value.find(d => d.id === datasetId);
      if (!dataset) {
        console.error('Dataset not found in local data');
        toast.add({ severity: 'error', summary: 'Error', detail: 'Dataset not found', life: 3000 });
        return;
      }
      const filename = dataset.filename;
      const response = await axios.get(`${API_BASE_URL}/trained_models/search_sort?dataset_filename=${filename}`);
      selectedDatasetTrainingResults.value = response.data;


    } catch (error) {
      console.error('Error fetching training results:', error);
       toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to fetch training results', life: 3000 });
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

  } catch (error) {
    console.error('Error downloading dataset:', error);
    toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to download dataset', life: 3000 });
  }

};

const formatDate = (dateString) => {
  if (!dateString) return '';
  return new Date(dateString).toLocaleDateString();
};
const goToTraining = (datasetId) => {
     router.push({ path: '/', query: { dataset: datasetId } });//add query parameter
};

const fileInfo = ref(null);
const previewData = ref([]);

const handleFileUpload = async (event) => {
  const file = event.files[0];
  if (!file) return;

  // File information
  fileInfo.value = {
    name: file.name,
    size: formatFileSize(file.size),
    rows: 0,
    columns: []
  };

  // Parse CSV
  const text = await file.text();
  parse(text, {
    header: true,
    preview: 10,
    skipEmptyLines: true,
    complete: (results) => {
      previewData.value = results.data;
      fileInfo.value = {
        ...fileInfo.value,
        rows: results.meta.cursor,
        columns: results.meta.fields || Object.keys(results.data[0] || {})
      };
      newDataset.file = file;
    },
    error: (error) => {
      console.error("CSV parsing error:", error);
      toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to parse CSV file', life: 5000 });
    }
  });
};

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const truncatedDescription = (text) => {
  const maxLines = 3;
  const lines = text.split('\n').slice(0, maxLines);
  return lines.join('\n') + (text.split('\n').length > maxLines ? '...' : '');
};

onMounted(fetchDatasets);
</script>

<style scoped>
.datasets-page {
  display: flex;
  flex-direction: column;
  align-items: center; /* Center items horizontally */
}

.add-dataset-section {
  margin-bottom: 1rem; /* Space between button and cards */
  align-self: flex-start;
}


.dataset-cards {
  display: flex;
  flex-wrap: wrap;
  justify-content: center; /* Center cards */
  gap: 1rem; /* Consistent spacing */
  max-width: 1200px;
}
.search-input{
    width: 50%;
    margin-bottom: 1rem;
}
.dataset-card {
    width: 300px; /* Fixed card width */
  cursor: pointer;
    border: 1px solid #ddd; /* Add a subtle border */
  transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth transition */
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.dataset-card:hover {
  transform: translateY(-5px); /* Lift the card slightly */
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* Increase shadow on hover */
}

.dataset-image {
  width: 100%;
  height: 200px; /* Fixed height for consistency */
  object-fit: cover;  /* Ensure image covers the area */
    border-bottom: 1px solid #ddd;
}
.dataset-description {
  text-align: left;
  padding: 0.5rem;

}

/* Dataset Details Dialog Styles */
.dataset-details {
  display: flex;
  gap: 20px; /* Space between image and info */
    padding: 20px;
    margin-bottom: 20px;
}

.dataset-details-image {
  max-width: 300px; /* Limit image size */
  max-height: 300px;
  object-fit: contain; /* Maintain aspect ratio */
    border: 1px solid #ddd;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dataset-info {
  text-align: left; /* Align text to the left */
}
.dataset-info p {
    margin: 0.5rem 0;
}

.dataset-info ul{
    padding-left: 20px;
}
.image-preview {
    max-width: 200px;
    max-height: 200px;
    margin-top: 10px;
}
.p-dialog .p-dialog-content{
  padding: 0 1.5rem 1.5rem 1.5rem;
}

/* Style for the training results section */
.training-results {
  margin-top: 2rem;
  text-align: left;
}
.preview-section {
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid #eee;
  border-radius: 4px;
}

.preview-section h4 {
  margin-bottom: 0.5rem;
}

.step-transition {
  transition: all 0.3s ease;
}

.image-preview {
  max-width: 200px;
  max-height: 200px;
  margin-top: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 4px;
}
.file-info-card {
  background: #f8f9fa;
  border-radius: 6px;
  padding: 1rem;
  margin-top: 1rem;
}

.file-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat-item {
  background: white;
  padding: 0.75rem;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-label {
  display: block;
  font-weight: 600;
  color: #495057;
  margin-bottom: 0.25rem;
  font-size: 0.9rem;
}

.stat-value {
  display: block;
  color: #6c757d;
  font-size: 0.95rem;
}

.preview-table {
  border: 1px solid #dee2e6;
  border-radius: 6px;
  overflow: hidden;
}

.preview-table h4 {
  background: #f8f9fa;
  padding: 0.75rem 1rem;
  margin: 0;
  border-bottom: 1px solid #dee2e6;
}

::v-deep .p-datatable {
  font-size: 0.9rem;
}

::v-deep .p-datatable-thead > tr > th {
  background: #f8f9fa !important;
}

::v-deep .p-datatable-tbody > tr > td {
  padding: 0.5rem 1rem !important;
}

.dataset-description {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.5em;
  max-height: 4.5em; /* 3 lines * 1.5em line height */
}

.no-description {
  color: #6c757d;
  font-style: italic;
}

/* Добавим стили для контейнера кнопок */
.action-buttons {
  display: flex;
  gap: 1rem; /* Отступ между кнопками */
  margin-top: 1.5rem;
  flex-wrap: wrap; /* Перенос на новую строку при нехватке места */
}
</style>
