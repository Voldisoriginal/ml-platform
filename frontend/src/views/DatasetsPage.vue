<template>
  <div class="datasets-page">
    <h1>Datasets</h1>

    <div class="controls-container">
      <div class="add-dataset-section">
        <Button label="Add Dataset" icon="pi pi-plus" @click="showAddDatasetDialog = true" />
      </div>
      <div class="search-section">
        <InputText v-model="searchQuery" placeholder="Search datasets (name, description)..." class="search-input" />
      </div>
    </div>


    <Dialog v-model:visible="showAddDatasetDialog" modal header="Add Dataset" :style="{ width: '70vw' }">
      <!-- Content of Add Dataset Dialog remains the same -->
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
              :maxFileSize="100000000"  /> <!-- Example: Limit file size -->
          </div>

          <div v-if="fileInfo" class="file-info-card">
              <!-- File Info Display remains the same -->
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
                    <span class="stat-label">Approx. Rows:</span>
                     <!-- Note: Papaparse preview might not give total rows accurately for large files -->
                    <span class="stat-value">{{ fileInfo.rows }} (preview)</span>
                    </div>
                    <div class="stat-item">
                    <span class="stat-label">Columns:</span>
                    <span class="stat-value">{{ fileInfo.columns.length }}</span>
                    </div>
                </div>
                 <div class="columns-list">
                    <strong>Columns:</strong> {{ fileInfo.columns.join(', ') }}
                </div>

                <div class="preview-table">
                    <h4>Preview (first 10 rows)</h4>
                    <DataTable :value="previewData" class="p-datatable-sm" scrollable scrollHeight="200px">
                    <Column v-for="col in fileInfo.columns" :key="col" :field="col" :header="col"></Column>
                    </DataTable>
                </div>
          </div>
          <Message v-if="uploadError" severity="error" :closable="false">{{ uploadError }}</Message>
        </div>

          <!-- Шаг 2: Детали датасета -->
          <div v-if="currentStep === 2">
              <!-- Fields for Name, Description, Author, Target, Image remain the same -->
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
                    v-model="newDataset.target_variable"
                    :options="fileInfo ? fileInfo.columns : []"
                    placeholder="Select target variable"
                    class="w-full"
                     required
                    />
                    <small v-if="!newDataset.target_variable && isFormValid === false" class="p-error">Target variable is required.</small>
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
        <!-- Footer buttons remain the same -->
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
          @click="proceedToStep2"
          :disabled="!newDataset.file || !!uploadError"
        />
        <Button
          v-if="currentStep === 2"
          label="Add Dataset"
          icon="pi pi-check"
          @click="addDataset"
          :disabled="!isFormValid || addingDataset"
          :loading="addingDataset"
          autofocus
        />
      </template>
    </Dialog>

    <!-- Dataset Cards -->
    <div v-if="loadingDatasets" class="loading-indicator">
        <ProgressSpinner />
        <p>Loading datasets...</p>
    </div>
    <div v-else-if="datasets.length === 0" class="no-datasets">
        <p>No datasets found. Add a new dataset to get started!</p>
    </div>
     <div class="dataset-cards" v-else>
         <!-- CHANGE: Click handler now navigates -->
        <Card v-for="dataset in filteredDatasets" :key="dataset.id" class="dataset-card" @click="navigateToDetail(dataset.id)">
          <template #header>
             <!-- Use getImageUrl helper -->
            <img :src="getImageUrl(dataset)" :alt="dataset.name || 'Dataset'" class="dataset-image"/>
          </template>
           <template #title>{{ dataset.name }}</template>
           <template #content>
            <!-- Use computed property for truncated description -->
             <p class="dataset-description" v-if="dataset.description">{{ truncatedDescription(dataset.description) }}</p>
             <p v-else class="no-description">No description provided</p>
           </template>
            <template #footer>
                <Button label="View Details" icon="pi pi-arrow-right" iconPos="right" class="p-button-sm p-button-text" />
            </template>
         </Card>
     </div>

     <!-- Removed the Details Dialog -->

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
import ProgressSpinner from 'primevue/progressspinner'; // Import spinner
import Message from 'primevue/message'; // Import message

const toast = useToast();
const router = useRouter();

// State for Datasets List Page
const datasets = ref([]);
const loadingDatasets = ref(true);
const searchQuery = ref('');

// State for Add Dataset Dialog
const showAddDatasetDialog = ref(false);
const currentStep = ref(1);
const fileUpload = ref(null);
const imageUpload = ref(null);
const fileInfo = ref(null);
const previewData = ref([]);
const uploadError = ref(null);
const addingDataset = ref(false); // Loading state for add button

const newDataset = reactive({
  name: '',
  description: '',
  author: '',
  target_variable: '',
  file: null,
  imageUrl: null, // For preview
  imageFile: null, // Actual image file
});

const API_BASE_URL = 'http://localhost:8000';

// Computed Properties
const isFormValid = computed(() => {
    // Step 1 check
    if (currentStep.value === 1) {
        return !!newDataset.file && !uploadError.value;
    }
    // Step 2 check
    if (currentStep.value === 2) {
         return !!newDataset.name && !!newDataset.target_variable && !!newDataset.file;
    }
    return false;
});

const filteredDatasets = computed(() => {
  if (!searchQuery.value) {
    return datasets.value;
  }
  const query = searchQuery.value.toLowerCase().trim();
  if (!query) {
      return datasets.value;
  }
  return datasets.value.filter(dataset =>
    (dataset.name && dataset.name.toLowerCase().includes(query)) ||
    (dataset.description && dataset.description.toLowerCase().includes(query))
    // Add || (dataset.author && dataset.author.toLowerCase().includes(query)) if needed
  );
});


// --- Methods for Datasets List ---

const fetchDatasets = async () => {
  loadingDatasets.value = true;
  try {
    const response = await axios.get(`${API_BASE_URL}/datasets/`);
     console.log("Fetched datasets:", response.data);
    if (Array.isArray(response.data)) {
        // Backend now provides `imageUrl` directly which points to the correct image endpoint
        datasets.value = response.data.map(ds => ({
            ...ds,
            // No need to construct URL here if backend sends full path or relative path like /dataset/.../image
            // imageUrl is already handled by the API response and getImageUrl helper
        }));
    } else {
        console.error("Unexpected response format from /datasets/:", response.data);
        toast.add({ severity: 'error', summary: 'Error', detail: 'Received unexpected data format.', life: 3000 });
        datasets.value = [];
    }
  } catch (error) {
    console.error('Error fetching datasets:', error);
    toast.add({ severity: 'error', summary: 'Error', detail: 'Failed to fetch datasets', life: 3000 });
    datasets.value = [];
  } finally {
    loadingDatasets.value = false;
  }
};

// Helper to get image URL (handles placeholder and backend URL)
const getImageUrl = (ds) => {
    if (ds?.imageUrl) {
        // Check if it's already an absolute URL
        if (ds.imageUrl.startsWith('http://') || ds.imageUrl.startsWith('https://')) {
            return ds.imageUrl;
        }
        // Assume relative path from API base
        return `${API_BASE_URL}${ds.imageUrl}`;
    }
    return `${API_BASE_URL}/placeholder.png`; // Fallback placeholder
};

const navigateToDetail = (datasetId) => {
  router.push({ name: 'DatasetDetail', params: { datasetId } });
};

const truncatedDescription = (text, maxLength = 100) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
};

// --- Methods for Add Dataset Dialog ---

const handleFileUpload = async (event) => {
  const file = event.files[0];
   uploadError.value = null; // Reset error on new file selection
   fileInfo.value = null;
   previewData.value = [];
   newDataset.file = null;

  if (!file) return;

  if (file.type !== 'text/csv') {
      uploadError.value = 'Invalid file type. Please select a CSV file.';
      toast.add({ severity: 'error', summary: 'Invalid File', detail: uploadError.value, life: 4000 });
       if (fileUpload.value) fileUpload.value.clear(); // Clear the FileUpload component
      return;
  }


  // Basic file info
   const baseInfo = {
        name: file.name,
        size: formatFileSize(file.size),
        rows: 'Calculating...', // Placeholder
        columns: []
    };
   fileInfo.value = baseInfo; // Show basic info immediately

  // Parse CSV using Papaparse
  try {
    const text = await file.text();
    parse(text, {
      header: true,
      preview: 10, // Only parse first 10 rows for preview
      skipEmptyLines: true,
      complete: (results) => {
          console.log("Parse results:", results);
         if (results.errors && results.errors.length > 0) {
             console.error("CSV parsing errors:", results.errors);
             uploadError.value = `Error parsing CSV: ${results.errors[0].message}. Please check file format.`;
             toast.add({ severity: 'error', summary: 'Parse Error', detail: uploadError.value, life: 5000 });
             fileInfo.value = null; // Clear info on error
             if (fileUpload.value) fileUpload.value.clear();
             return;
         }
        if (!results.data || results.data.length === 0 || !results.meta.fields || results.meta.fields.length === 0) {
             uploadError.value = 'CSV file appears to be empty or has no header row.';
             toast.add({ severity: 'error', summary: 'Empty File', detail: uploadError.value, life: 5000 });
             fileInfo.value = null;
             if (fileUpload.value) fileUpload.value.clear();
             return;
         }

        previewData.value = results.data;
        fileInfo.value = {
          ...baseInfo,
          // Note: results.meta.cursor gives rows parsed in preview, not total rows
          rows: results.meta.cursor, // This is approximate based on preview
          columns: results.meta.fields || []
        };
        newDataset.file = file; // Assign file only if parse is successful
         newDataset.name = file.name.replace('.csv', ''); // Pre-fill name
         newDataset.target_variable = ''; // Reset target variable selection

      },
      error: (error) => {
        console.error("CSV parsing fatal error:", error);
        uploadError.value = `Failed to parse CSV file: ${error.message}`;
        toast.add({ severity: 'error', summary: 'Parse Error', detail: uploadError.value, life: 5000 });
         fileInfo.value = null; // Clear info on error
         if (fileUpload.value) fileUpload.value.clear();
      }
    });
  } catch (readError) {
      console.error("Error reading file:", readError);
      uploadError.value = `Error reading file: ${readError.message}`;
      toast.add({ severity: 'error', summary: 'File Read Error', detail: uploadError.value, life: 5000 });
      fileInfo.value = null;
      if (fileUpload.value) fileUpload.value.clear();
  }
};

const proceedToStep2 = () => {
    if (isFormValid.value) {
        currentStep.value = 2;
    } else {
        toast.add({ severity: 'warn', summary: 'Missing File', detail: 'Please select a valid CSV file first.', life: 3000 });
    }
};


const handleImageUpload = (event) => {
  const file = event.files[0];
    if (file){
      // Optional: Check file size or type for images too
      newDataset.imageFile = file;
      newDataset.imageUrl = URL.createObjectURL(file); // Create preview URL
    } else {
      newDataset.imageFile = null;
      newDataset.imageUrl = null;
    }
};

const addDataset = async () => {
  if (!isFormValid.value) {
       toast.add({ severity: 'warn', summary: 'Incomplete Form', detail: 'Please fill in all required fields (Name, Target Variable).', life: 3000 });
    return;
  }

  addingDataset.value = true; // Set loading state

  const formData = new FormData();
  formData.append('file', newDataset.file);
  formData.append('name', newDataset.name);
  if (newDataset.description) formData.append('description', newDataset.description);
  if (newDataset.author) formData.append('author', newDataset.author);
  formData.append('target_variable', newDataset.target_variable); // Always send target
  if (newDataset.imageFile) formData.append('image', newDataset.imageFile);


  try {
    const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    toast.add({severity: 'success', summary: 'Success', detail: 'Dataset added successfully!', life: 3000});
    resetForm();
    showAddDatasetDialog.value = false;
    fetchDatasets(); // Refresh the list

  } catch (error) {
     console.error('Error adding dataset:', error);
     const detail = error.response?.data?.detail || 'Failed to add dataset. Check server logs.';
     toast.add({severity: 'error', summary: 'Error Adding Dataset', detail: detail, life: 5000});
  } finally {
      addingDataset.value = false; // Reset loading state
  }
};

const cancelAddDataset = () => {
    resetForm();
    showAddDatasetDialog.value = false;
};

const resetForm = () => {
    currentStep.value = 1;
    if (fileUpload.value) fileUpload.value.clear();
    if (imageUpload.value) imageUpload.value.clear();

    newDataset.name = '';
    newDataset.description = '';
    newDataset.author = '';
    newDataset.target_variable = '';
    newDataset.file = null;
    newDataset.imageUrl = null;
    newDataset.imageFile = null;

    fileInfo.value = null;
    previewData.value = [];
    uploadError.value = null;
    addingDataset.value = false;
};


const formatFileSize = (bytes) => {
  if (bytes === 0 || !bytes) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// --- Lifecycle ---
onMounted(fetchDatasets);

</script>

<style scoped>
.datasets-page {
  padding: 2rem;
  max-width: 1400px; /* Wider max-width maybe */
  margin: 0 auto;
}

.controls-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  flex-wrap: wrap; /* Allow wrapping on smaller screens */
  gap: 1rem;
}

.add-dataset-section {
  flex-shrink: 0; /* Prevent button from shrinking */
}

.search-section {
  flex-grow: 1; /* Allow search to take available space */
  min-width: 250px; /* Minimum width for search */
}

.search-input {
  width: 100%; /* Make input take full width of its container */
   max-width: 500px; /* Optional max width */
}

.dataset-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); /* Responsive grid */
  gap: 1.5rem;
}

.dataset-card {
  cursor: pointer;
  border: 1px solid #e0e0e0;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  display: flex; /* Use flex for better control over content */
  flex-direction: column; /* Stack header, content, footer */
}

.dataset-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.dataset-image {
  width: 100%;
  height: 180px; /* Fixed height */
  object-fit: cover;
  border-bottom: 1px solid #eee;
}

/* Ensure content area grows */
.dataset-card :deep(.p-card-content) {
  flex-grow: 1;
  padding-top: 0.75rem; /* Adjust padding */
  padding-bottom: 0.75rem;
}

.dataset-card :deep(.p-card-title) {
  font-size: 1.15rem; /* Slightly larger title */
  margin-bottom: 0.5rem;
}

.dataset-description {
  font-size: 0.9rem;
  color: #555;
  line-height: 1.4;
  /* Clamping text to N lines */
  display: -webkit-box;
  -webkit-line-clamp: 3; /* Show 3 lines */
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
   min-height: calc(1.4em * 3); /* Ensure space for 3 lines */
}

.no-description {
  font-size: 0.9rem;
  color: #888;
  font-style: italic;
   min-height: calc(1.4em * 3); /* Ensure space for 3 lines */
}

.dataset-card :deep(.p-card-footer) {
  padding-top: 0.75rem;
  border-top: 1px solid #eee;
  text-align: right; /* Align button to the right */
}

/* Add Dataset Dialog Styles */
.file-info-card {
  background: #f8f9fa;
  border-radius: 6px;
  padding: 1rem;
  margin-top: 1rem;
  border: 1px solid #dee2e6;
}

.file-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.75rem 1rem;
  margin-bottom: 1rem;
}
.columns-list {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 1rem;
    word-break: break-word;
}

.stat-item {
  background: white;
  padding: 0.5rem 0.75rem;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  font-size: 0.9rem;
}

.stat-label {
  display: block;
  font-weight: 600;
  color: #495057;
  margin-bottom: 0.25rem;
  font-size: 0.85rem;
}

.stat-value {
  display: block;
  color: #6c757d;
}

.preview-table {
  border: 1px solid #dee2e6;
  border-radius: 6px;
  overflow: hidden; /* Needed for scrollable */
  margin-top: 1rem;
}

.preview-table h4 {
  background: #e9ecef;
  padding: 0.5rem 1rem;
  margin: 0;
  border-bottom: 1px solid #dee2e6;
   font-size: 0.95rem;
   font-weight: 600;
}

.image-preview {
  max-width: 200px;
  max-height: 200px;
  margin-top: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 4px;
  display: block; /* Ensure it behaves like a block element */
}

.p-field {
    margin-bottom: 1.5rem; /* Consistent spacing between fields */
}

/* Loading and No Data States */
.loading-indicator, .no-datasets {
    text-align: center;
    margin-top: 3rem;
    color: #6c757d;
}
.loading-indicator p {
    margin-top: 0.5rem;
    font-size: 1.1rem;
}

.no-datasets p {
    font-size: 1.2rem;
}

</style>