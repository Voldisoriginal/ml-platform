<template>
  <div class="inference-page">
    <Toast />
    <h1>Model Inference</h1>

    <div v-if="loading" class="loading-spinner">
      <ProgressSpinner />
      <p>Loading model details...</p>
    </div>

    <div v-if="error" class="error-message">
      <Message severity="error">{{ error }}</Message>
      <Button label="Go Back" icon="pi pi-arrow-left" @click="goBack" class="p-button-sm p-button-secondary mt-2" />
    </div>

    <div v-if="modelDetails && !loading && !error" class="content-grid">
      <!-- Section 1: Model & Dataset Info -->
      <Card class="info-card">
        <template #title>Model & Dataset Information</template>
        <template #content>
          <div class="info-item"><strong>Model ID:</strong> {{ modelDetails.id }}</div>
          <div class="info-item"><strong>Model Type:</strong> {{ modelDetails.model_type }}</div>
          <div class="info-item"><strong>Trained On:</strong> {{ modelDetails.dataset_filename }}</div>
          <div class="info-item"><strong>Target Variable:</strong> {{ modelDetails.target_column }}</div>
          <div class="info-item"><strong>Training Date:</strong> {{ formatDate(modelDetails.start_time) }}</div>
          <div v-if="datasetDetails" class="info-item">
            <strong>Dataset Name:</strong> {{ datasetDetails.name || 'N/A' }}
          </div>
          <!-- Add more dataset details if needed -->
        </template>
      </Card>

      <!-- Section 2: Metrics -->
      <Card class="metrics-card">
        <template #title>Model Metrics</template>
        <template #content>
          <DataTable :value="formattedMetrics" class="p-datatable-sm" responsiveLayout="scroll">
            <Column field="metric" header="Metric"></Column>
            <Column field="value" header="Value">
              <template #body="slotProps">
                {{ formatMetricValue(slotProps.data.value) }}
              </template>
            </Column>
          </DataTable>
        </template>
      </Card>

      <!-- Section 3: Inference Control & Status -->
      <Card class="control-card">
        <template #title>Inference Control</template>
        <template #content>
          <div class="status-section">
            <strong>Status:</strong>
            <Tag :severity="statusSeverity" :value="inferenceStatus" class="ml-2" />
          </div>
          <div class="action-buttons mt-3">
            <Button
              label="Start Inference"
              icon="pi pi-play"
              @click="startInference"
              :disabled="inferenceStatus === 'running' || loadingStatus"
              :loading="loadingStatus"
              class="p-button-success p-button-sm mr-2"
            />
            <Button
              label="Stop Inference"
              icon="pi pi-stop-circle"
              @click="stopInference"
              :disabled="inferenceStatus !== 'running' || loadingStatus"
              :loading="loadingStatus"
              class="p-button-danger p-button-sm"
            />
            <Button
              icon="pi pi-refresh"
              @click="fetchInferenceStatus"
              :loading="loadingStatus"
              class="p-button-secondary p-button-sm ml-2"
              v-tooltip.top="'Refresh Status'"
            />
          </div>
           <Message v-if="inferenceStatus !== 'running'" severity="warn" :closable="false" class="mt-3">
              Inference service is not running. Start it to enable predictions.
           </Message>
        </template>
      </Card>

      <!-- Section 4: Prediction Interface -->
      <Card class="prediction-card" v-if="inferenceStatus === 'running'">
        <template #title>Make Predictions</template>
        <template #content>
          <TabView>
            <!-- Tab 1: JSON Input -->
            <TabPanel header="JSON Input">
              <p>Enter feature values for a single prediction:</p>
              <div v-if="features.length > 0" class="p-fluid grid formgrid">
                <div v-for="feature in features" :key="feature" class="field col-12 md:col-6">
                  <label :for="`input-${feature}`">{{ feature }}</label>
                  <InputText :id="`input-${feature}`" v-model="predictionInput[feature]" type="text" />
                  <!-- Consider using InputNumber for numeric features if type info is available -->
                </div>
              </div>
              <div v-else class="text-center p-3">
                 <ProgressSpinner style="width:30px; height:30px" strokeWidth="8" v-if="loadingFeatures"/>
                 <span v-else>Could not load features.</span>
              </div>
              <div class="mt-3 text-right">
                <Button
                  label="Predict (JSON)"
                  icon="pi pi-send"
                  @click="predictDirectInput"
                  :disabled="!canPredictDirect || isPredicting"
                  :loading="isPredicting"
                />
              </div>
            </TabPanel>

            <!-- Tab 2: CSV File Input -->
            <TabPanel header="CSV File Upload">
              <p>Upload a CSV file for batch predictions. The CSV must contain the same columns (excluding the target variable: '{{modelDetails.target_column}}') used for training.</p>
                <FileUpload
                  ref="csvUploader"
                  mode="basic"
                  name="predict_file"
                  accept=".csv"
                  :maxFileSize="10000000"
                  :customUpload="true"
                  @uploader="handleFileUpload"
                  :auto="true"
                  chooseLabel="Upload CSV & Predict"
                  :disabled="isPredicting"
                />
                <small class="p-error block mt-1" v-if="fileUploadError">{{ fileUploadError }}</small>

             </TabPanel>
          </TabView>

          <!-- Prediction Output -->
          <div v-if="predictionOutput !== null || predictionError" class="output-section mt-4">
            <h4>Prediction Result:</h4>
            <div v-if="predictionError" class="error-message">
               <Message severity="error" :closable="false">{{ predictionError }}</Message>
            </div>
             <ScrollPanel v-else style="width: 100%; height: 200px" class="output-textarea">
                <pre>{{ formattedPredictionOutput }}</pre>
             </ScrollPanel>
          </div>
           <div v-if="isPredicting" class="text-center mt-3">
               <ProgressSpinner style="width:40px; height:40px" strokeWidth="6"/>
               <p>Predicting...</p>
           </div>

        </template>
      </Card>
       <Card class="prediction-card" v-else>
         <template #title>Make Predictions</template>
          <template #content>
             <Message severity="info" :closable="false">
                Start the inference service to enable predictions.
            </Message>
          </template>
       </Card>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, reactive } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import { useToast } from 'primevue/usetoast';
import Papa from 'papaparse'; // For CSV parsing

// PrimeVue Components
import Card from 'primevue/card';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Button from 'primevue/button';
import Tag from 'primevue/tag';
import ProgressSpinner from 'primevue/progressspinner';
import Message from 'primevue/message';
import InputText from 'primevue/inputtext';
// import InputNumber from 'primevue/inputnumber'; // Optional for numeric inputs
import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';
import FileUpload from 'primevue/fileupload';
import Textarea from 'primevue/textarea';
import Toast from 'primevue/toast';
import ScrollPanel from 'primevue/scrollpanel';
import Tooltip from 'primevue/tooltip';


const API_BASE_URL = 'http://localhost:8000'; // Adjust if needed

// Props & Router
const props = defineProps({
  modelId: {
    type: String,
    required: true
  }
});
const route = useRoute();
const router = useRouter();
const toast = useToast();

// Reactive State
const modelDetails = ref(null);
const datasetDetails = ref(null); // Optional: To store more dataset info
const features = ref([]);
const loadingFeatures = ref(false);
const inferenceStatus = ref('unknown'); // 'running', 'not running', 'unknown'
const loading = ref(true);
const loadingStatus = ref(false); // For start/stop/refresh buttons
const error = ref(null);
const predictionInput = reactive({}); // For JSON input form
const predictionOutput = ref(null);
const predictionError = ref(null);
const isPredicting = ref(false); // Loading state for prediction calls
const fileUploadError = ref(null);
const csvUploader = ref(null); // Ref for FileUpload component

// Computed Properties
const formattedMetrics = computed(() => {
  if (!modelDetails.value || !modelDetails.value.metrics) return [];
  return Object.entries(modelDetails.value.metrics).map(([key, value]) => ({
    metric: key,
    value: value
  }));
});

const statusSeverity = computed(() => {
  switch (inferenceStatus.value) {
    case 'running': return 'success';
    case 'not running': return 'danger';
    default: return 'warning';
  }
});

// Check if all feature inputs have values for direct prediction
const canPredictDirect = computed(() => {
    if (features.value.length === 0) return false;
    return features.value.every(feature =>
        predictionInput[feature] !== undefined && predictionInput[feature] !== null && predictionInput[feature] !== ''
    );
});

const formattedPredictionOutput = computed(() => {
    if (predictionOutput.value === null) return '';
    try {
        // Try to pretty-print if it's JSON, otherwise return as is
        return JSON.stringify(predictionOutput.value, null, 2);
    } catch {
        return String(predictionOutput.value);
    }
});

// Methods
const fetchData = async () => {
  loading.value = true;
  error.value = null;
  try {
    // 1. Fetch Model Details
    // Using search_sort endpoint with model_id filter as it returns the necessary TrainedModel structure
    const modelResponse = await axios.get(`${API_BASE_URL}/trained_models/search_sort`, {
      params: { model_id: props.modelId }
    });

    if (modelResponse.data && modelResponse.data.length > 0) {
      modelDetails.value = modelResponse.data[0];
      // Initialize prediction input object based on fetched features later
    } else {
      throw new Error(`Model with ID ${props.modelId} not found.`);
    }

    // 2. Fetch Features required by the model
    await fetchFeatures();


    // 3. Fetch Dataset Details (Optional, if needed beyond filename/target)
    // await fetchDatasetDetails(modelDetails.value.dataset_id); // Assuming dataset_id is available

    // 4. Fetch Initial Inference Status
    await fetchInferenceStatus();

  } catch (err) {
    console.error("Error fetching model data:", err);
    error.value = `Failed to load model details: ${err.response?.data?.detail || err.message}`;
    toast.add({ severity: 'error', summary: 'Loading Error', detail: error.value, life: 5000 });
  } finally {
    loading.value = false;
  }
};

const fetchFeatures = async () => {
    if (!modelDetails.value) return;
    loadingFeatures.value = true;
    try {
        const response = await axios.get(`${API_BASE_URL}/features/${props.modelId}`);
        features.value = response.data || [];
        // Initialize predictionInput keys
        predictionInput.value = {}; // Reset
        features.value.forEach(feature => {
            predictionInput[feature] = ''; // Initialize with empty string or null
        });
    } catch (err) {
        console.error("Error fetching features:", err);
        toast.add({ severity: 'warn', summary: 'Feature Loading Failed', detail: 'Could not load model input features.', life: 3000 });
        features.value = [];
    } finally {
       loadingFeatures.value = false;
    }
};

const fetchInferenceStatus = async () => {
  loadingStatus.value = true;
  try {
    const response = await axios.get(`${API_BASE_URL}/inference_status/${props.modelId}`);
    inferenceStatus.value = response.data.status || 'unknown';
  } catch (err) {
    console.error("Error fetching inference status:", err);
    inferenceStatus.value = 'unknown'; // Set to unknown on error
    // Don't show toast for simple refresh usually, unless it's a persistent failure
  } finally {
    loadingStatus.value = false;
  }
};

const startInference = async () => {
  loadingStatus.value = true;
  predictionOutput.value = null; // Clear previous output
  predictionError.value = null;
  try {
    await axios.post(`${API_BASE_URL}/start_inference/${props.modelId}`);
    inferenceStatus.value = 'running';
    toast.add({ severity: 'success', summary: 'Success', detail: 'Inference service started.', life: 3000 });
  } catch (err) {
    console.error("Error starting inference:", err);
    const detail = err.response?.data?.detail || 'Failed to start inference service.';
    toast.add({ severity: 'error', summary: 'Error', detail: detail, life: 5000 });
    fetchInferenceStatus(); // Refresh status in case of failure
  } finally {
    loadingStatus.value = false;
  }
};

const stopInference = async () => {
  loadingStatus.value = true;
  predictionOutput.value = null; // Clear previous output
  predictionError.value = null;
  try {
    await axios.delete(`${API_BASE_URL}/stop_inference/${props.modelId}`);
    inferenceStatus.value = 'not running';
    toast.add({ severity: 'success', summary: 'Success', detail: 'Inference service stopped.', life: 3000 });
  } catch (err) {
    console.error("Error stopping inference:", err);
    const detail = err.response?.data?.detail || 'Failed to stop inference service.';
    toast.add({ severity: 'error', summary: 'Error', detail: detail, life: 5000 });
    fetchInferenceStatus(); // Refresh status in case of failure
  } finally {
    loadingStatus.value = false;
  }
};

const predictDirectInput = async () => {
  if (!canPredictDirect.value) {
      toast.add({ severity: 'warn', summary: 'Input Missing', detail: 'Please fill in all feature values.', life: 3000 });
      return;
  }

  isPredicting.value = true;
  predictionOutput.value = null;
  predictionError.value = null;

  // Prepare data in the format expected by the backend: List[Dict]
  const inputData = [{}];
   features.value.forEach(feature => {
      // Attempt to convert to number if possible, otherwise keep as string
      const value = predictionInput[feature];
      inputData[0][feature] = isNaN(Number(value)) || value === '' ? value : Number(value);
  });


  try {
    const response = await axios.post(`${API_BASE_URL}/predict/${props.modelId}`, inputData);
    predictionOutput.value = response.data.predictions;
    toast.add({ severity: 'success', summary: 'Prediction Successful', detail: 'Received prediction results.', life: 3000 });
  } catch (err) {
    console.error("Error during direct prediction:", err);
    predictionError.value = `Prediction failed: ${err.response?.data?.detail || err.message}`;
    toast.add({ severity: 'error', summary: 'Prediction Error', detail: predictionError.value, life: 5000 });
  } finally {
    isPredicting.value = false;
  }
};

const handleFileUpload = async (event) => {
  const file = event.files[0];
  if (!file) return;

  isPredicting.value = true;
  predictionOutput.value = null;
  predictionError.value = null;
  fileUploadError.value = null;

  try {
    const fileContent = await readFileContent(file);
    const parseResult = Papa.parse(fileContent, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true, // Automatically convert numbers/booleans
    });

    if (parseResult.errors.length > 0) {
      console.error("CSV Parsing Errors:", parseResult.errors);
      throw new Error(`CSV Parsing Error: ${parseResult.errors[0].message} on row ${parseResult.errors[0].row + 1}`); // +1 for 1-based index
    }

    const requiredFeatures = new Set(features.value);
    const csvHeaders = new Set(parseResult.meta.fields);

    // Check if all required features are present in the CSV
    for (const feature of requiredFeatures) {
        if (!csvHeaders.has(feature)) {
            throw new Error(`Missing required column in CSV: '${feature}'`);
        }
    }

    // Optional: Check for extra columns (could warn user)
    // const extraCols = [...csvHeaders].filter(h => !requiredFeatures.has(h) && h !== modelDetails.value?.target_column);
    // if (extraCols.length > 0) {
    //     console.warn("CSV contains extra columns:", extraCols);
    // }


    // Prepare data: Remove target column if present, ensure only feature columns are sent
    const dataToSend = parseResult.data.map(row => {
        const filteredRow = {};
        features.value.forEach(feature => {
             if(row.hasOwnProperty(feature)) { // Check if feature exists in row data
                filteredRow[feature] = row[feature];
             } else {
                 // Handle missing value for a required feature if necessary (e.g., set to null or raise specific error)
                 console.warn(`Row missing value for required feature: ${feature}. Setting to null/undefined.`);
                 filteredRow[feature] = null; // Or handle as error
             }
        });
        return filteredRow;
    });


    if (dataToSend.length === 0) {
        throw new Error("CSV file is empty or contains no valid data rows.");
    }

    // Send data to backend
    const response = await axios.post(`${API_BASE_URL}/predict/${props.modelId}`, dataToSend);
    predictionOutput.value = response.data.predictions;
    toast.add({ severity: 'success', summary: 'Batch Prediction Successful', detail: `Processed ${dataToSend.length} rows.`, life: 3000 });

  } catch (err) {
    console.error("Error during file upload prediction:", err);
    predictionError.value = `File prediction failed: ${err.response?.data?.detail || err.message}`;
    fileUploadError.value = predictionError.value; // Show error near upload button
    toast.add({ severity: 'error', summary: 'Prediction Error', detail: predictionError.value, life: 6000 });
  } finally {
    isPredicting.value = false;
    // Reset the file input visually if possible (depends on PrimeVue version/behavior)
    if (csvUploader.value) {
       // csvUploader.value.clear(); // Might work in some versions
    }
  }
};

const readFileContent = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(new Error("Failed to read file"));
    reader.readAsText(file); // Read as text for PapaParse
  });
};

const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  try {
    return new Date(dateString).toLocaleString();
  } catch (e) {
    return dateString; // Return original if parsing fails
  }
};

const formatMetricValue = (value) => {
    if (typeof value === 'number') {
        return value.toFixed(4); // Adjust precision as needed
    }
    return value;
};

const goBack = () => {
  router.go(-1); // Go back to the previous page
  // Or redirect to a specific page like router.push('/running-models');
};

// Lifecycle Hook
onMounted(() => {
  fetchData();
});

// Directives (for Tooltip)
const vTooltip = Tooltip;

</script>

<style scoped>
.inference-page {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

h1 {
  text-align: center;
  margin-bottom: 2rem;
  color: var(--primary-color);
}

.loading-spinner, .error-message {
  text-align: center;
  margin-top: 3rem;
}

.content-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
}

.p-card {
  border: 1px solid var(--surface-d);
  border-radius: 8px;
  overflow: hidden; /* Ensures content respects card boundaries */
}

.p-card .p-card-title {
  font-size: 1.2rem;
  border-bottom: 1px solid var(--surface-d);
  padding-bottom: 0.8rem;
  margin-bottom: 1rem;
}
.p-card .p-card-content {
   padding-top: 0; /* Remove default top padding if title has bottom border */
}


.info-card .info-item {
  margin-bottom: 0.75rem;
  font-size: 0.95rem;
}

.info-card strong {
  color: var(--text-color-secondary);
  margin-right: 0.5rem;
}

.metrics-card .p-datatable-sm {
  font-size: 0.9rem;
}

.control-card .status-section {
  display: flex;
  align-items: center;
  font-size: 1.1rem;
}

.control-card .p-tag {
  font-weight: bold;
}

.control-card .action-buttons .p-button {
  margin-right: 0.5rem; /* Ensure spacing between buttons */
}

.prediction-card .p-tabview .p-tabview-panels {
  padding: 1.5rem 0 0 0; /* Add padding top to panels */
}

.prediction-card .p-tabview-nav-link {
    font-size: 0.95rem;
}


.field label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  font-size: 0.9rem;
}

.output-section {
  margin-top: 2rem;
  border: 1px solid var(--surface-d);
  border-radius: 6px;
  padding: 1rem;
  background-color: var(--surface-a);
}

.output-section h4 {
  margin-top: 0;
  margin-bottom: 1rem;
  color: var(--text-color-secondary);
}

.output-textarea {
    background-color: var(--surface-b); /* Slightly different background for pre */
    border-radius: 4px;
    padding: 0.5rem 1rem;
}

.output-textarea pre {
  white-space: pre-wrap;       /* CSS3 */
  white-space: -moz-pre-wrap;  /* Mozilla */
  white-space: -pre-wrap;      /* Opera 4-6 */
  white-space: -o-pre-wrap;    /* Opera 7 */
  word-wrap: break-word;       /* Internet Explorer 5.5+ */
  margin: 0;
  font-family: var(--font-family-monospace); /* Use monospace font */
  font-size: 0.85rem;
  color: var(--text-color);
}


/* Make file upload button more prominent if needed */
::v-deep(.p-fileupload-basic .p-button) {
  width: 100%;
  justify-content: center;
}

.mt-1 { margin-top: 0.25rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-3 { margin-top: 1rem; }
.mt-4 { margin-top: 1.5rem; }
.ml-2 { margin-left: 0.5rem; }
.mr-2 { margin-right: 0.5rem; }
.text-right { text-align: right; }
.text-center { text-align: center; }
.block { display: block; }


@media (max-width: 768px) {
  .content-grid {
    grid-template-columns: 1fr; /* Stack cards on smaller screens */
  }
   .p-fluid.grid.formgrid .field {
       width: 100%; /* Full width inputs on small screens */
   }
}

</style>