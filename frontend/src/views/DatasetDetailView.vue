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
          icon="pi pi-cog"
          @click="goToTrain(dataset.id)"
          class="p-button-outlined p-button-secondary"
          v-tooltip.bottom="'Go to Training Setup'"
        />
        <Button
          label="Download"
          icon="pi pi-download"
          @click="downloadDataset(dataset.filename)"
          class="p-button-outlined p-button-secondary"
          v-tooltip.bottom="'Download CSV'"
        />
        <Button
          icon="pi pi-ellipsis-v"
          class="p-button-text p-button-secondary"
          @click="toggleShareMenu"
          aria-haspopup="true"
          aria-controls="share_menu"
          v-tooltip.bottom="'More Actions'"
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

    <!-- БЛОК: Визуализации -->
    <div class="visualization-section">
      <h2>Data Visualization</h2>

      <!-- Кнопка запуска -->
      <div v-if="!visualizationData && !isLoadingVisualizations && !visualizationError" class="start-visualization">
         <p>Generate interactive charts to explore the dataset's numerical features.</p>
        <Button
          label="Visualize Data"
          icon="pi pi-chart-bar"
          @click="startVisualization"
          :loading="isStartingVisualization"
          class="p-button-raised p-button-primary"
        />
      </div>

      <!-- Индикатор загрузки -->
      <div v-if="isLoadingVisualizations" class="loading-container visualization-loading">
        <ProgressSpinner style="width:50px;height:50px" strokeWidth="8" />
        <p>Generating visualizations... This may take a moment.</p>
        <small v-if="currentVisualizationTaskId">(Task ID: {{ currentVisualizationTaskId }})</small>
      </div>

      <!-- Сообщение об ошибке -->
      <div v-if="visualizationError && !isLoadingVisualizations" class="error-container visualization-error">
        <Message severity="error" :closable="false">
          Failed to generate or load visualizations: {{ visualizationError }}
        </Message>
         <Button
          label="Retry Visualization"
          icon="pi pi-refresh"
          @click="startVisualization"
          :loading="isStartingVisualization"
          class="p-button-raised p-button-warning p-button-sm"
          style="margin-top: 1rem;"
        />
      </div>

      <!-- Табы с графиками -->
      <TabView v-if="visualizationData && !isLoadingVisualizations && !visualizationError" v-model:activeIndex="activeVisualizationTabIndex" class="visualization-tabs p-tabview-cards">
        <!-- Histograms Tab -->
        <TabPanel header="Histograms">
          <div v-if="!visualizationData.histograms || visualizationData.histograms.length === 0" class="no-data-message">
              <i class="pi pi-info-circle" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i>
              <p>No numerical columns with sufficient variance found for histograms.</p>
          </div>
          <div v-else class="charts-grid">
            <Card v-for="(hist, index) in preparedHistograms" :key="'hist-' + index" class="chart-card">
              <template #title>{{ hist.column }}</template>
              <template #content>
                <Chart type="bar" :data="hist.chartData" :options="hist.chartOptions" />
              </template>
            </Card>
          </div>
        </TabPanel>

        <!-- Box Plots Tab -->
        <TabPanel header="Box Plots">
           <div v-if="!visualizationData.boxplots || visualizationData.boxplots.length === 0" class="no-data-message">
               <i class="pi pi-info-circle" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i>
               <p>No numerical columns with sufficient data found for box plot statistics.</p>
           </div>
           <div v-else>
              <!-- View Mode Toggle -->
              <div class="view-toggle-container">
                <SelectButton v-model="boxplotViewMode" :options="boxplotViewOptions" optionLabel="label" optionValue="value" aria-labelledby="boxplot-view-mode" />
              </div>

              <!-- Statistics View -->
              <div v-if="boxplotViewMode === 'stats'" class="charts-grid stats-view">
                  <Card v-for="(boxplot, index) in visualizationData.boxplots" :key="'box-stat-' + index" class="chart-card data-card-simple">
                      <template #title>{{ boxplot.column }}</template>
                      <template #content>
                          <ul class="stats-list">
                              <li><span class="stat-label">Min:</span> <span class="stat-value">{{ boxplot.min?.toFixed(2) ?? 'N/A' }}</span></li>
                              <li><span class="stat-label">Q1 (25%):</span> <span class="stat-value">{{ boxplot.q1?.toFixed(2) ?? 'N/A' }}</span></li>
                              <li><span class="stat-label">Median (50%):</span> <span class="stat-value">{{ boxplot.median?.toFixed(2) ?? 'N/A' }}</span></li>
                              <li><span class="stat-label">Q3 (75%):</span> <span class="stat-value">{{ boxplot.q3?.toFixed(2) ?? 'N/A' }}</span></li>
                              <li><span class="stat-label">Max:</span> <span class="stat-value">{{ boxplot.max?.toFixed(2) ?? 'N/A' }}</span></li>
                          </ul>
                      </template>
                  </Card>
              </div>

              <!-- Chart View (Individual Charts) -->
              <div v-if="boxplotViewMode === 'chart'" class="charts-grid boxplot-chart-grid">
                 <Card v-for="(boxplotChart, index) in preparedIndividualBoxplots" :key="'box-chart-' + index" class="chart-card">
                      <template #title>{{ boxplotChart.column }}</template>
                      <template #content>
                          <Chart type="boxplot" :data="boxplotChart.chartData" :options="boxplotChart.chartOptions" />
                      </template>
                  </Card>
              </div>
           </div>
        </TabPanel>

        <!-- Correlation Matrix Tab -->
        <TabPanel header="Correlation Matrix">
          <div v-if="!visualizationData.correlation_matrix || !visualizationData.correlation_matrix.columns || visualizationData.correlation_matrix.columns.length < 2" class="no-data-message">
               <i class="pi pi-info-circle" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i>
               <p>Correlation matrix requires at least two numerical columns.</p>
          </div>
          <div v-else>
               <!-- View Mode Toggle -->
               <div class="view-toggle-container">
                  <SelectButton v-model="correlationViewMode" :options="correlationViewOptions" optionLabel="label" optionValue="value" aria-labelledby="correlation-view-mode" />
               </div>

               <!-- Table View -->
               <div v-if="correlationViewMode === 'table'" class="table-view correlation-table-container">
                 <DataTable :value="preparedCorrelationData" responsiveLayout="scroll" class="p-datatable-sm correlation-table">
                     <Column field="column" header="" style="font-weight: bold; min-width: 120px; background-color: #f8f9fa; border-right: 1px solid #dee2e6; position: sticky; left: 0; z-index: 1;"></Column>
                     <Column v-for="colName in visualizationData.correlation_matrix.columns" :key="colName" :field="colName" :header="colName">
                         <template #body="slotProps">
                            <span :style="getCorrelationCellStyle(slotProps.data[colName])" v-tooltip.top="`Correlation(${slotProps.data.column}, ${colName}): ${formatCorrelationValue(slotProps.data[colName])}`">
                                {{ formatCorrelationValue(slotProps.data[colName]) }}
                            </span>
                         </template>
                     </Column>
                 </DataTable>
               </div>

               <!-- Chart View (Heatmap) -->
               <div v-if="correlationViewMode === 'chart'" class="chart-container heatmap-chart-container">
                 <Card class="chart-card single-chart-card">
                    <template #title>Correlation Heatmap</template>
                    <template #content>
                        <Chart type="matrix" :data="preparedCorrelationMatrixChartData" :options="correlationMatrixChartOptions" />
                    </template>
                 </Card>
               </div>
          </div>
        </TabPanel>
      </TabView>

    </div>
    <!-- КОНЕЦ БЛОКА ВИЗУАЛИЗАЦИИ -->


    <!-- Основной TabView с Data Card / Training Results / Running Services -->
    <TabView v-model:activeIndex="activeTabIndex" class="main-tabs">
      <!-- Data Card Tab -->
      <TabPanel header="Data Card">
        <div class="data-card-content">
          <div class="data-card-grid">
            <div class="info-item"> <label>Author:</label> <span>{{ dataset.author || 'N/A' }}</span> </div>
            <div class="info-item"> <label>Target Variable:</label> <Tag :value="dataset.target_variable || 'Not Set'" severity="info" /> </div>
            <div class="info-item"> <label>File Name:</label> <span>{{ dataset.filename }}</span> </div>
            <div class="info-item"> <label>File Size:</label> <span>{{ dataset.file_size !== null && dataset.file_size >= 0 ? formatFileSize(dataset.file_size) : (dataset.file_size === -1 ? 'File Missing' : 'N/A') }}</span> </div>
            <div class="info-item"> <label>Rows:</label> <span>{{ dataset.row_count !== null && dataset.row_count >= 0 ? dataset.row_count : (dataset.row_count === -1 ? 'File Missing' : 'N/A') }}</span> </div>
            <div class="info-item"> <label>Columns Count:</label> <span>{{ dataset.columns ? dataset.columns.length : 'N/A' }}</span> </div>
          </div>
          <h3>Columns</h3>
          <DataTable v-if="columnInfo && columnInfo.length > 0" :value="columnInfo" responsiveLayout="scroll" class="p-datatable-sm p-datatable-striped">
              <Column field="name" header="Name" :sortable="true"></Column>
              <Column field="type" header="Detected Type" :sortable="true"></Column>
          </DataTable>
          <p v-else>Column type information not available.</p>
          <h3>Data Preview (First 20 Rows)</h3>
          <ProgressSpinner v-if="loadingPreview" style="width:50px;height:50px" strokeWidth="8" />
          <div v-else-if="previewError" class="error-message"> Could not load data preview: {{ previewError }} </div>
          <DataTable v-else-if="filePreview && filePreview.length > 0" :value="filePreview" responsiveLayout="scroll" scrollable scrollHeight="400px" class="p-datatable-sm preview-table-detail">
              <Column v-for="col in dataset.columns" :key="col" :field="col" :header="col"></Column>
          </DataTable>
          <p v-else-if="!loadingPreview && !previewError">No data preview available or file is empty.</p>
        </div>
      </TabPanel>

      <!-- Training Results Tab -->
      <TabPanel header="Training Results">
         <ProgressSpinner v-if="loadingResults" style="width:50px;height:50px" strokeWidth="8" />
          <div v-else-if="trainingResultsError" class="error-message"> Could not load training results: {{ trainingResultsError }} </div>
         <div v-else>
             <p v-if="!trainingResults || trainingResults.length === 0" class="no-data-message"> <i class="pi pi-info-circle" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i> <br> No training results found for this dataset yet. </p>
            <DataTable v-else :value="trainingResults" responsiveLayout="scroll" class="p-datatable-striped" sortField="start_time" :sortOrder="-1">
                <Column field="model_type" header="Model Type" :sortable="true"></Column>
                <Column field="target_column" header="Target" :sortable="true"> <template #body="{data}"> <Tag :value="data.target_column" /> </template> </Column>
                <Column header="Metrics"> <template #body="{data}"> <div class="metrics-list"> <span v-for="(value, key) in data.metrics" :key="key" class="metric-chip"> {{ key }}: {{ value?.toFixed ? value.toFixed(3) : value }} </span> </div> </template> </Column> <!-- Добавил проверку на null -->
                <Column field="start_time" header="Trained On" :sortable="true"> <template #body="{data}"> {{ formatDateTime(data.start_time) }} </template> </Column>
                <Column header="Actions"> <template #body="{data}"> <Button icon="pi pi-chart-line" class="p-button-sm p-button-info p-button-text" @click="showModelDetails(data)" v-tooltip.top="'View Training Details'" /> <Button icon="pi pi-play" class="p-button-sm p-button-success p-button-text" @click="startInference(data.id)" v-tooltip.top="'Start Inference Service'" :disabled="isModelRunning(data.id)" /> </template> </Column>
             </DataTable>
         </div>
        </TabPanel>

      <!-- *** НОВАЯ ВКЛАДКА: Running Services *** -->
      <TabPanel header="Running Services">
         <!-- Индикатор загрузки -->
         <ProgressSpinner v-if="loadingRunningModels" style="width:50px;height:50px" strokeWidth="8" />
         <!-- Сообщение об ошибке загрузки -->
         <div v-else-if="runningModelsError" class="error-message"> Could not load running services status: {{ runningModelsError }} </div>
         <!-- Сообщение, если нет запущенных сервисов для этого датасета -->
         <div v-else-if="!runningModelsForThisDataset || runningModelsForThisDataset.length === 0" class="no-data-message">
              <i class="pi pi-info-circle" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i> <br>
              No inference services are currently running for models trained on this dataset.
          </div>
         <!-- Таблица с запущенными сервисами -->
         <DataTable v-else :value="runningModelsForThisDataset" responsiveLayout="scroll" class="p-datatable-striped p-datatable-sm">
            <Column field="model_type" header="Model Type" :sortable="true"></Column>
            <Column field="target_column" header="Target" :sortable="true">
                 <template #body="{data}"> <Tag :value="data.target_column" /> </template>
            </Column>
            <!-- Можно добавить метрики, если они полезны здесь
             <Column header="Metrics"> <template #body="{data}"> <div class="metrics-list"> <span v-for="(value, key) in data.metrics" :key="key" class="metric-chip"> {{ key }}: {{ value?.toFixed ? value.toFixed(3) : value }} </span> </div> </template> </Column> -->
            <Column header="API Endpoint">
                <template #body="{data}">
                    <code>/predict/{{ data.model_id }}</code>
                 </template>
             </Column>
            <Column header="Actions" style="min-width: 10rem;">
                <template #body="{data}">
                     <!-- Кнопка для перехода к документации API -->
                    <Button
                        icon="pi pi-book"
                        class="p-button-sm p-button-secondary p-button-text"
                        @click="goToApiDocs(data.model_id)"
                        v-tooltip.top="'View API Docs (Swagger)'"
                     />
                     <!-- Кнопка для остановки сервиса -->
                    <Button
                        icon="pi pi-stop-circle"
                        class="p-button-sm p-button-danger p-button-text"
                        @click="stopInferenceService(data.model_id)"
                        :loading="isStoppingService[data.model_id]"
                        v-tooltip.top="'Stop Inference Service'"
                     />
                 </template>
             </Column>
         </DataTable>
      </TabPanel>
      <!-- *** КОНЕЦ НОВОЙ ВКЛАДКИ *** -->

    </TabView>

    <!-- Model Details Dialog -->
    <Dialog v-model:visible="showDetailsDialog" header="Model Training Details" :modal="true" :style="{ width: '60vw', minWidth: '400px' }" :breakpoints="{'960px': '75vw', '640px': '90vw'}" dismissableMask >
      <div v-if="selectedModel" class="model-details-dialog">
        <div class="detail-grid">
            <div class="detail-item"><label>Model ID:</label> <span>{{ selectedModel.id }}</span></div>
            <div class="detail-item"><label>Model Type:</label> <span>{{ selectedModel.model_type }}</span></div>
            <div class="detail-item"><label>Task Type:</label> <span>{{ selectedModel.task_type || 'N/A' }}</span></div>
            <div class="detail-item"><label>Dataset:</label> <span>{{ formatFilename(selectedModel.dataset_filename) }}</span></div>
            <div class="detail-item"><label>Target Column:</label> <span>{{ selectedModel.target_column }}</span></div>
            <div class="detail-item"><label>Training Date:</label> <span>{{ formatDateTime(selectedModel.start_time) }}</span></div>
            <div class="detail-item"><label>Train Size:</label> <span>{{ selectedModel.train_settings?.train_size ?? 'N/A' }}</span></div>
            <div class="detail-item"><label>Random State:</label> <span>{{ selectedModel.train_settings?.random_state ?? 'N/A' }}</span></div>
        </div>
         <div v-if="selectedModel.params && Object.keys(selectedModel.params).length > 0" class="params-section">
          <h3>Model Parameters</h3> <pre>{{ JSON.stringify(selectedModel.params, null, 2) }}</pre>
        </div>
        <div v-else class="params-section"> <h3>Model Parameters</h3> <p><i>No specific parameters recorded for this model type.</i></p> </div>
         <h3>Metrics</h3>
        <div class="metrics-grid-dialog">
          <div v-for="(value, key) in selectedModel.metrics" :key="key" class="metric-card-dialog" >
            <div class="metric-title-dialog">{{ key }}</div> <div class="metric-value-dialog">{{ value?.toFixed ? value.toFixed(4) : value }}</div> <!-- Добавил проверку на null -->
          </div>
        </div>
      </div>
      <template #footer> <Button label="Close" icon="pi pi-times" @click="showDetailsDialog = false" class="p-button-text" /> </template>
    </Dialog>

  </div>
  <div v-else-if="loadingDataset" class="loading-container main-loading"> <ProgressSpinner /> <p>Loading dataset details...</p> </div>
   <div v-else-if="error" class="error-container main-error"> <Message severity="error" :closable="false">{{ error }}</Message> <Button label="Go Back to Datasets" icon="pi pi-arrow-left" @click="goBack" class="p-button-secondary" style="margin-top: 1rem;"/> </div>
   <Toast />
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue';
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
import Chart from 'primevue/chart';
import Card from 'primevue/card';
import Toast from 'primevue/toast';
import SelectButton from 'primevue/selectbutton';
import Tooltip from 'primevue/tooltip'; // Import directive

// Directives need to be registered if not done globally
const vTooltip = Tooltip;

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const route = useRoute();
const router = useRouter();
const toast = useToast();

// --- State ---
const dataset = ref(null);
const trainingResults = ref([]);
const filePreview = ref([]);
const columnInfo = ref([]);
const runningModels = ref([]); // Global list of all running models
const loadingDataset = ref(true);
const loadingResults = ref(false);
const loadingPreview = ref(false);
const loadingRunningModels = ref(false); // Loading state for running models
const error = ref(null);
const trainingResultsError = ref(null);
const previewError = ref(null);
const runningModelsError = ref(null); // Error state for running models
const activeTabIndex = ref(0); // Index for main tabs
const shareMenu = ref();
const showDetailsDialog = ref(false);
const selectedModel = ref(null);
const isStoppingService = ref({}); // { model_id: true/false }

const datasetId = computed(() => route.params.datasetId);

// --- State for Visualization ---
const visualizationData = ref(null); // { histograms: [], boxplots: [], correlation_matrix: {} }
const isLoadingVisualizations = ref(false);
const isStartingVisualization = ref(false); // Separate flag for button loading state
const visualizationError = ref(null);
const activeVisualizationTabIndex = ref(0); // Index for visualization tabs
const pollingIntervalId = ref(null); // ID for polling interval
const currentVisualizationTaskId = ref(null); // ID of the running/pending visualization task for *this* dataset

// --- State for View Toggles ---
const boxplotViewMode = ref('stats'); // 'stats' or 'chart'
const boxplotViewOptions = ref([
    {label: 'Statistics', value: 'stats'},
    {label: 'Chart', value: 'chart'}
]);
const correlationViewMode = ref('table'); // 'table' or 'chart'
const correlationViewOptions = ref([
    {label: 'Table', value: 'table'},
    {label: 'Heatmap', value: 'chart'}
]);

// --- Menu Items ---
const shareMenuItems = ref([
  {
    label: 'Share Link',
    icon: 'pi pi-share-alt',
    command: () => shareDataset()
  },
  // TODO: Add Delete Dataset option here?
]);

// --- Methods ---

const goToTrain = (datasetId)=>{
  router.push({ path: '/', query: { dataset: datasetId } });
};

const fetchDatasetDetails = async (id) => {
  loadingDataset.value = true;
  error.value = null;
  // Reset all states
  dataset.value = null;
  filePreview.value = [];
  columnInfo.value = [];
  trainingResults.value = [];
  runningModels.value = [];
  visualizationData.value = null;
  visualizationError.value = null;
  isLoadingVisualizations.value = false;
  runningModelsError.value = null;
  previewError.value = null;
  trainingResultsError.value = null;
  stopPolling(); // Stop any previous visualization polling

  try {
    const response = await axios.get(`${API_BASE_URL}/dataset/${id}`);
    dataset.value = response.data;

     // Process column types
     if (dataset.value.column_types) {
         columnInfo.value = Object.entries(dataset.value.column_types).map(([name, type]) => ({ name, type }));
     } else if (dataset.value.columns) {
         columnInfo.value = dataset.value.columns.map(name => ({ name, type: 'Unknown' }));
     }

    if (dataset.value?.filename) {
      // Fetch related data concurrently
      await Promise.all([
        fetchTrainingResults(dataset.value.filename),
        fetchDatasetPreview(dataset.value.filename),
        fetchRunningModels(), // Fetch *all* running models globally
        fetchVisualizations() // Fetch visualization status/data *for this* dataset
      ]);
    } else if (!dataset.value) {
         error.value = "Received empty dataset details from server.";
    }

  } catch (err) {
    console.error('Error fetching dataset details:', err);
    if (err.response && err.response.status === 404) {
        error.value = `Dataset with ID ${id} not found.`;
    } else {
        error.value = `Failed to load dataset details: ${err.response?.data?.detail || err.message}`;
    }
    dataset.value = null; // Ensure dataset is null on error
  } finally {
    loadingDataset.value = false;
  }
};

const fetchTrainingResults = async (filename) => {
  loadingResults.value = true;
  trainingResultsError.value = null;
  trainingResults.value = [];
  try {
    const response = await axios.get(`${API_BASE_URL}/trained_models/search_sort`, {
        params: {
            dataset_filename: filename,
            sort_by: 'start_time',
            sort_order: 'desc'
        }
    });
    trainingResults.value = response.data;
  } catch (err) {
    console.error('Error fetching training results:', err);
    trainingResultsError.value = `Failed to load training results: ${err.response?.data?.detail || err.message}`;
  } finally {
    loadingResults.value = false;
  }
};

const fetchDatasetPreview = async (filename) => {
  loadingPreview.value = true;
  previewError.value = null;
  filePreview.value = [];
  try {
    const response = await axios.get(`${API_BASE_URL}/dataset_preview/${filename}`, {
        params: { rows: 20 }
    });
    filePreview.value = response.data.preview;
     // Update column types from preview if more accurate/available
     if(response.data.column_types) {
         columnInfo.value = Object.entries(response.data.column_types).map(([name, type]) => ({ name, type }));
     }
  } catch (err) {
    console.error('Error fetching data preview:', err);
     if (err.response?.status === 404) {
         previewError.value = 'Dataset file not found for preview.';
     } else {
        previewError.value = `Failed to load data preview: ${err.response?.data?.detail || err.message}`;
     }
  } finally {
    loadingPreview.value = false;
  }
};

// Fetches ALL running models
const fetchRunningModels = async () => {
    loadingRunningModels.value = true;
    runningModelsError.value = null;
    try {
        const response = await axios.get(`${API_BASE_URL}/running_models/`);
        runningModels.value = response.data || []; // Ensure it's an array
    } catch (error) {
        console.error('Failed to fetch running models status:', error);
        runningModelsError.value = `Could not fetch running model status: ${error.message}`;
        runningModels.value = []; // Clear on error
    } finally {
        loadingRunningModels.value = false;
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
    // Try to extract original filename (after UUID_)
    const originalFilename = filename.includes('_') ? filename.split('_').slice(1).join('_') : filename;
    link.setAttribute('download', originalFilename || filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    toast.add({ severity: 'success', summary: 'Success', detail: 'Dataset download started.', life: 3000 });
  } catch (err) {
    console.error('Error downloading dataset:', err);
    toast.add({ severity: 'error', summary: 'Error', detail: `Failed to download dataset: ${err.response?.data?.detail || err.message}`, life: 5000 });
  }
};

const shareDataset = () => {
  const url = window.location.href;
  navigator.clipboard.writeText(url).then(() => {
    toast.add({ severity: 'success', summary: 'Success', detail: 'Link copied to clipboard!', life: 3000 });
  }, (err) => {
    console.error('Could not copy text: ', err);
    toast.add({ severity: 'warn', summary: 'Warning', detail: 'Could not copy link automatically.', life: 5000 });
  });
};

const toggleShareMenu = (event) => {
  shareMenu.value.toggle(event);
};

// Used in Training Results tab
const startInference = async (modelId) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/start_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Started',
      detail: `Model ${modelId} inference service started. Status: ${response.data.status}`,
      life: 4000
    });
    // Refresh running models status after a short delay
    setTimeout(fetchRunningModels, 1000);
  } catch (error) {
    console.error('Error starting inference:', error);
    toast.add({
      severity: 'error',
      summary: 'Error Starting Inference',
      detail: `Failed to start inference: ${error.response?.data?.detail || error.message}`,
      life: 5000
    });
  }
};

// Used in Running Services tab
const stopInferenceService = async (modelId) => {
    if (!modelId) return;
    isStoppingService.value[modelId] = true; // Show loader on button
    try {
        await axios.delete(`${API_BASE_URL}/stop_inference/${modelId}`);
        toast.add({
            severity: 'success',
            summary: 'Service Stopped',
            detail: `Inference service for model ${modelId} stopped.`,
            life: 3000
        });
        // Refresh the list of running models
        await fetchRunningModels();
    } catch (error) {
         console.error('Error stopping inference service:', error);
         toast.add({
            severity: 'error',
            summary: 'Error Stopping Service',
            detail: `Failed to stop service: ${error.response?.data?.detail || error.message}`,
            life: 5000
        });
    } finally {
         isStoppingService.value[modelId] = false; // Hide loader
    }
};

// Used in Running Services tab
const goToApiDocs = (modelId) => {
    if (!modelId) return;
    // Construct URL for Swagger UI - adjust path if your setup is different
    const swaggerPath = `/docs#/default/predict_endpoint_predict__model_id__post`.replace('{model_id}', modelId);
    // Use API_BASE_URL, removing potential trailing /api/v1 or similar if needed
    const baseUrl = API_BASE_URL.replace(/\/api(\/v\d+)?$/, '');
    const fullUrl = `${baseUrl}${swaggerPath}`;
    window.open(fullUrl, '_blank'); // Open in new tab
};


const showModelDetails = (model) => {
  selectedModel.value = model;
  showDetailsDialog.value = true;
};

const goBack = () => {
    router.push({ name: 'Datasets' }); // Assumes a route named 'Datasets'
}

// Used in Training Results tab to disable "Start" button
const isModelRunning = (modelId) => {
    return runningModels.value.some(m => m.model_id === modelId && m.status === 'running');
};

// --- Visualization Methods ---

const fetchVisualizations = async () => {
  if (!datasetId.value) return;

  visualizationError.value = null;
  // Don't set isLoadingVisualizations initially, let polling or final status control it.

  try {
    const response = await axios.get(`${API_BASE_URL}/datasets/${datasetId.value}/visualizations`);
    const vizRecord = response.data; // This is the VisualizationDataResponse or null

    if (vizRecord) {
        console.log("Fetched visualization record:", vizRecord);
        const taskIdFromRecord = vizRecord.celery_task_id;

        if (vizRecord.status === 'SUCCESS') {
            visualizationData.value = vizRecord.visualization_data;
            visualizationError.value = null;
            isLoadingVisualizations.value = false; // Success, hide loader
            stopPolling();
            currentVisualizationTaskId.value = null; // Task completed
            boxplotViewMode.value = 'stats';
            correlationViewMode.value = 'table';
        } else if (vizRecord.status === 'FAILURE') {
            visualizationError.value = vizRecord.error_message || 'Unknown error during previous visualization attempt.';
            visualizationData.value = null;
            isLoadingVisualizations.value = false; // Failure, hide loader
            stopPolling();
            currentVisualizationTaskId.value = null; // Task completed (failed)
            console.error("Found failed visualization attempt:", vizRecord);
        } else if (vizRecord.status === 'PENDING') {
            visualizationData.value = null; // Clear old data
            visualizationError.value = null;
            isLoadingVisualizations.value = true; // Show loader
            if (taskIdFromRecord) {
                console.log(`Found PENDING visualization with Task ID: ${taskIdFromRecord}. Starting/continuing polling.`);
                currentVisualizationTaskId.value = taskIdFromRecord; // Set as the task to poll
                startPolling(taskIdFromRecord);
            } else {
                 console.error("Found PENDING visualization record but no Task ID. Cannot poll status.");
                 visualizationError.value = "Visualization is pending, but its task ID is missing. Please retry.";
                 isLoadingVisualizations.value = false;
                 stopPolling();
                 currentVisualizationTaskId.value = null;
            }
        } else {
            // Unknown status
            console.warn(`Visualization record found with unexpected status: ${vizRecord.status}.`);
            visualizationData.value = null;
            visualizationError.value = null;
            isLoadingVisualizations.value = false;
            stopPolling();
            currentVisualizationTaskId.value = null;
        }
    } else {
         // No visualization record found
         console.log("No existing visualization record found for this dataset.");
         visualizationData.value = null;
         visualizationError.value = null;
         isLoadingVisualizations.value = false;
         stopPolling();
         currentVisualizationTaskId.value = null;
    }
  } catch (error) {
    console.error('Error fetching visualization data:', error);
    if (error.response && error.response.status === 404) {
         console.log("No visualization record found (404).");
         visualizationData.value = null;
         visualizationError.value = null;
    } else {
        visualizationError.value = `Could not load visualization status: ${error.message}`;
        visualizationData.value = null;
    }
    isLoadingVisualizations.value = false;
    stopPolling();
    currentVisualizationTaskId.value = null;
  }
};


const startVisualization = async () => {
  if (!datasetId.value) return;

  isStartingVisualization.value = true;
  isLoadingVisualizations.value = true; // Show main loader
  visualizationError.value = null;
  visualizationData.value = null;
  stopPolling(); // Stop previous polling if any

  try {
    const response = await axios.post(`${API_BASE_URL}/datasets/${datasetId.value}/visualize`);
    const newTaskId = response.data.task_id;
    currentVisualizationTaskId.value = newTaskId; // Set the NEW task ID
    toast.add({ severity: 'info', summary: 'Processing', detail: 'Visualization generation started...', life: 3000 });
    startPolling(newTaskId); // Start polling for the NEW task

  } catch (error) {
    console.error('Error starting visualization task:', error);
    visualizationError.value = `Failed to start visualization: ${error.response?.data?.detail || error.message}`;
    isLoadingVisualizations.value = false; // Hide loader on startup error
    stopPolling();
    currentVisualizationTaskId.value = null; // Reset task ID on failure to start
    toast.add({ severity: 'error', summary: 'Error', detail: visualizationError.value, life: 5000 });
  } finally {
    isStartingVisualization.value = false; // Remove button loader
  }
};

const pollVisualizationStatus = async (taskId) => {
  // Ensure we are still supposed to poll this task
  if (!taskId || taskId !== currentVisualizationTaskId.value || !pollingIntervalId.value) {
      console.log(`Polling check: Stop condition met. TaskId=${taskId}, Current=${currentVisualizationTaskId.value}, IntervalId=${pollingIntervalId.value}`);
      // Don't stop polling here necessarily, the check in setInterval will handle it if task ID changed
      if (!pollingIntervalId.value) { // Stop only if interval was cleared elsewhere
          console.log("Polling interval already cleared.");
      }
      return;
  }

  console.log(`Polling status for task ${taskId}...`);
  try {
    const response = await axios.get(`${API_BASE_URL}/visualization_status/${taskId}`);
    const { status, error: errorInfo, result } = response.data;

    console.log(`Task ${taskId} status: ${status}`);

    // Check again if the task ID is still the current one *after* the API call returns
    if (taskId !== currentVisualizationTaskId.value) {
        console.log(`Polling result received for ${taskId}, but current task is now ${currentVisualizationTaskId.value}. Ignoring result.`);
        return; // Avoid race conditions if user clicked "Retry" quickly
    }

    if (status === 'SUCCESS') {
      stopPolling();
      toast.add({ severity: 'success', summary: 'Success', detail: 'Visualizations generated!', life: 3000 });
      // Fetch the final data using the main fetch function which handles UI state
      await fetchVisualizations(); // This will set data, hide loader, clear task ID

    } else if (status === 'FAILURE') {
      stopPolling();
      let errorMessage = 'Visualization task failed.';
      if (errorInfo) {
          errorMessage = `${errorInfo.type || 'Error'}: ${errorInfo.message || 'Unknown reason'}`;
          console.error('Visualization task failed:', errorInfo);
          // if (errorInfo.traceback) console.error("Traceback:\n", errorInfo.traceback);
      }
      visualizationError.value = errorMessage;
      isLoadingVisualizations.value = false; // Hide loader
      currentVisualizationTaskId.value = null; // Task completed (failed)
      toast.add({ severity: 'error', summary: 'Visualization Failed', detail: errorMessage, life: 6000 });

    } else if (status === 'PENDING' || status === 'STARTED' || status === 'RETRY') {
      // Task still running, ensure loader is visible
      isLoadingVisualizations.value = true;
    } else {
         // Unknown status
         console.warn(`Unknown task status received: ${status}. Stopping poll.`);
         stopPolling();
         visualizationError.value = `Received an unexpected status: ${status}`;
         isLoadingVisualizations.value = false;
         currentVisualizationTaskId.value = null;
    }
  } catch (err) {
    console.error('Error polling visualization status:', err);
    // Check if task ID changed during error handling
     if (taskId !== currentVisualizationTaskId.value) {
        console.log(`Polling error occurred for ${taskId}, but current task is now ${currentVisualizationTaskId.value}. Ignoring error.`);
        return;
    }

    if (err.response && err.response.status === 404) {
        console.error(`Task ${taskId} not found during polling (404). Stopping poll.`);
        stopPolling();
        visualizationError.value = `Visualization task ${taskId} could not be found. It may have expired or failed. Please try again.`;
        isLoadingVisualizations.value = false;
        currentVisualizationTaskId.value = null; // Clear invalid task ID
    } else {
         // Other network errors - show warning, but let polling continue
         toast.add({ severity: 'warn', summary: 'Polling Error', detail: 'Could not check visualization status. Retrying...', life: 2000 });
    }
  }
};


const startPolling = (taskId) => {
  if (!taskId) return;
  stopPolling(); // Clear previous interval just in case

  console.log(`Starting polling for visualization task: ${taskId}`);
  // currentVisualizationTaskId is already set by the caller (startVisualization or fetchVisualizations)
  isLoadingVisualizations.value = true; // Show loader

  // Immediate check after 1s
  setTimeout(() => {
      if (taskId === currentVisualizationTaskId.value) { // Check if still relevant
          pollVisualizationStatus(taskId);
      }
  }, 1000);

  // Set interval
  pollingIntervalId.value = setInterval(() => {
      if (taskId === currentVisualizationTaskId.value) { // Check if still relevant
          pollVisualizationStatus(taskId);
      } else {
          // Task ID changed, stop this specific interval
          console.log(`Polling interval: Task ID changed from ${taskId} to ${currentVisualizationTaskId.value}. Stopping interval ${pollingIntervalId.value}.`);
          clearInterval(pollingIntervalId.value); // Stop this interval instance
           // Check if the stopped interval was the 'active' one
          if (pollingIntervalId.value === Number(pollingIntervalId.value)) { // A bit hacky check if it's the one we stored
             pollingIntervalId.value = null;
          }
      }
  }, 5000);
};

const stopPolling = () => {
  if (pollingIntervalId.value !== null) {
    clearInterval(pollingIntervalId.value);
    console.log(`Polling interval ${pollingIntervalId.value} cleared.`);
    pollingIntervalId.value = null;
  }
  // Don't change isLoadingVisualizations or currentVisualizationTaskId here
};


// --- Computed Properties ---

const runningModelsForThisDataset = computed(() => {
    if (!dataset.value || !dataset.value.filename || !runningModels.value) {
        return [];
    }
    // Filter the global list for models matching the current dataset's filename
    return runningModels.value.filter(model => model.dataset_filename === dataset.value.filename && model.status === 'running');
});

const getImageUrl = (ds) => {
    if (ds?.imageUrl) {
        if (ds.imageUrl.startsWith('http://') || ds.imageUrl.startsWith('https://')) {
            return ds.imageUrl;
        }
        if (ds.imageUrl.startsWith('/')) {
             return `${API_BASE_URL}${ds.imageUrl}`;
        }
         return `${API_BASE_URL}/dataset/${ds.id}/image`;
    }
    return `${API_BASE_URL}/placeholder.png`;
};

const formattedDate = (dateString) => {
  if (!dateString) return 'N/A';
  try {
      return new Date(dateString).toLocaleDateString(undefined, {
          year: 'numeric', month: 'long', day: 'numeric'
      });
  } catch (e) {
      return 'Invalid Date';
  }
};

const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    try {
        return new Date(dateString).toLocaleString(undefined, {
            year: 'numeric', month: 'numeric', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
     } catch (e) {
      return 'Invalid DateTime';
    }
};

const formatFileSize = (bytes) => {
  if (bytes === null || bytes === undefined || bytes < 0) return 'N/A';
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const index = Math.min(i, sizes.length - 1); // Ensure index is within bounds
  // Handle potential division by zero or log(0) if index calculation goes wrong
  if (index < 0) return 'N/A';
  return parseFloat((bytes / Math.pow(k, index)).toFixed(2)) + ' ' + sizes[index];
};

const formatFilename = (filename) => {
    // Removes UUID prefix if present
    return filename?.replace(/^[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}_/, '') || filename || 'N/A';
};

// --- Chart Data Preparation (Computed) ---

const preparedHistograms = computed(() => {
  if (!visualizationData.value?.histograms) return [];
  // ... (keep existing implementation)
  return visualizationData.value.histograms.map(hist => {
    const labels = hist.bins.slice(0, -1).map((edge, i) => {
       const nextEdge = hist.bins[i+1];
       const formatEdge = (val) => val.toFixed(val % 1 !== 0 ? 2 : 0);
       return `[${formatEdge(edge)}, ${formatEdge(nextEdge)})`;
    });
    return {
      column: hist.column,
      chartData: {
        labels: labels,
        datasets: [
          {
            label: 'Frequency',
            data: hist.counts,
            backgroundColor: '#A0C4FF',
            borderColor: '#4A90E2',
            borderWidth: 1,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
          }
        ]
      },
      chartOptions: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Frequency' } },
          x: { title: { display: true, text: 'Value Bins' }, ticks: { maxRotation: 45, minRotation: 0 } }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    title: (tooltipItems) => `Bin: ${tooltipItems[0].label}`,
                    label: (context) => ` Frequency: ${context.parsed.y}`
                }
            }
        }
      }
    };
  });
});


const preparedIndividualBoxplots = computed(() => {
    const boxplotData = visualizationData.value?.boxplots;
    if (!boxplotData || boxplotData.length === 0) {
        return [];
    }

    const defaultChartOptions = {
        responsive: true,
        maintainAspectRatio: false, // Important for fitting in card
        scales: {
            y: {
                beginAtZero: false, // Box plots don't necessarily start at zero
                title: { display: true, text: 'Value' }
            },
            x: {
                // Hide x-axis title, label is enough
                title: { display: false },
            }
        },
        plugins: {
            legend: { display: false },
            title: { display: false }, // Use card title
            tooltip: {
                callbacks: {
                    // Tooltip for a single boxplot item
                    label: function(context) {
                        // Ensure datasetIndex and dataIndex are valid
                        if (context.datasetIndex == null || context.dataIndex == null) return '';
                        const item = context.chart.data.datasets[context.datasetIndex]?.data[context.dataIndex];
                        if (!item) return '';
                        return [
                            `Max: ${item.max?.toFixed(2) ?? 'N/A'}`,
                            `Q3: ${item.q3?.toFixed(2) ?? 'N/A'}`,
                            `Median: ${item.median?.toFixed(2) ?? 'N/A'}`,
                            `Q1: ${item.q1?.toFixed(2) ?? 'N/A'}`,
                            `Min: ${item.min?.toFixed(2) ?? 'N/A'}`,
                        ];
                    }
                }
            }
        }
    };

    return boxplotData.map(bp => {
        const singleBoxplotDataItem = {
            min: bp.min ?? undefined,
            q1: bp.q1 ?? undefined,
            median: bp.median ?? undefined,
            q3: bp.q3 ?? undefined,
            max: bp.max ?? undefined,
        };

        return {
            column: bp.column,
            chartData: {
                labels: [bp.column], // Label for the single category on x-axis
                datasets: [{
                    label: bp.column, // Dataset label (can be same as column)
                    data: [singleBoxplotDataItem], // Data MUST be an array for boxplot type
                    backgroundColor: 'rgba(160, 196, 255, 0.5)',
                    borderColor: '#4A90E2',
                    borderWidth: 1,
                    itemRadius: 3,
                    itemStyle: 'circle', // Or 'rect', etc. for median point
                    outlierStyle: 'circle',
                    // padding: 10 // Might add too much space for single plot
                }]
            },
            // Use a deep copy of default options if you might modify them later per chart
            // For now, using the same options object is fine as it's not modified dynamically here
            chartOptions: defaultChartOptions
        };
    });
});


const preparedCorrelationData = computed(() => {
  const matrix = visualizationData.value?.correlation_matrix;
  if (!matrix || !matrix.columns || !matrix.data || matrix.columns.length !== matrix.data.length) {
       if (matrix && (!matrix.data || matrix.columns?.length !== matrix.data?.length)) {
            console.warn("Correlation matrix data and columns length mismatch.");
       }
       return [];
  }

  return matrix.data.map((row, rowIndex) => {
    if (rowIndex >= matrix.columns.length || row.length !== matrix.columns.length) {
         console.warn(`Correlation matrix row ${rowIndex} length mismatch or index out of bounds.`);
         return null;
    };
    const rowData = { column: matrix.columns[rowIndex] };
    row.forEach((value, colIndex) => {
       if (colIndex < matrix.columns.length) {
           rowData[matrix.columns[colIndex]] = value;
       }
    });
    return rowData;
  }).filter(Boolean); // Remove nulls if any row had issues
});


const preparedCorrelationMatrixChartData = computed(() => {
  const matrix = visualizationData.value?.correlation_matrix;
  if (!matrix || !matrix.columns || !matrix.data || matrix.columns.length < 2) {
    return { datasets: [] };
  }
  // Add validation for matrix dimensions
  if (matrix.columns.length !== matrix.data.length || !matrix.data.every(row => row.length === matrix.columns.length)) {
      console.error("Correlation matrix dimensions are inconsistent.");
      return { datasets: [] };
  }

  const labels = matrix.columns;
  const data = [];
  matrix.data.forEach((row, i) => {
    const yLabel = labels[i];
    row.forEach((value, j) => {
       const xLabel = labels[j];
       // Ensure value is a valid number for the heatmap
       if (value !== null && value !== undefined && !isNaN(value)) {
         data.push({ x: xLabel, y: yLabel, v: value });
       } else {
         // Optionally represent NaN/null differently, e.g., with v: null or skip
         data.push({ x: xLabel, y: yLabel, v: null }); // Use null for coloring
       }
    });
  });

  return {
      datasets: [{
          label: 'Correlation',
          data: data,
          backgroundColor: function(context) {
             const value = context.dataset.data[context.dataIndex]?.v;
             return getColorForValue(value); // Handles null/NaN internally
          },
          borderColor: 'rgba(0,0,0,0.1)',
          borderWidth: 1,
          width: ({chart}) => Math.max(10, (chart.chartArea?.width ?? 300) / labels.length - 1), // Ensure min width
          height: ({chart}) => Math.max(10, (chart.chartArea?.height ?? 300) / labels.length - 1), // Ensure min height
      }]
  };
});

const getColorForValue = (value) => {
    if (value === null || value === undefined || isNaN(value)) return 'rgba(200, 200, 200, 0.5)'; // Grey for invalid/null

    // Scale: Strong Red (-1) -> White (0) -> Strong Blue (+1)
    const intensity = Math.abs(value);
    let r, g, b;
    const whitePoint = 255;

    if (value > 0) { // Positive correlation (Blue scale)
        r = Math.round(whitePoint * (1 - value));
        g = Math.round(whitePoint * (1 - value));
        b = whitePoint;
    } else { // Negative correlation (Red scale)
        r = whitePoint;
        g = Math.round(whitePoint * (1 + value));
        b = Math.round(whitePoint * (1 + value));
    }

    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));

    return `rgb(${r}, ${g}, ${b})`;
};


const correlationMatrixChartOptions = computed(() => {
   const labels = visualizationData.value?.correlation_matrix?.columns ?? [];
   if (!labels.length) return {}; // Return empty options if no labels

   return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'category',
                labels: labels,
                position: 'bottom',
                ticks: { display: true, autoSkip: false, maxRotation: 90, minRotation: 45 },
                grid: { display: false }
            },
            y: {
                type: 'category',
                labels: labels,
                position: 'left',
                offset: true,
                ticks: { display: true, autoSkip: false },
                 grid: { display: false },
                reverse: true
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    title: function() { return ''; },
                    label: function(context) {
                        const item = context.dataset.data[context.dataIndex];
                        if (!item) return '';
                        const valueFormatted = (item.v === null || item.v === undefined) ? 'N/A' : item.v.toFixed(3);
                        return `Corr(${item.y}, ${item.x}): ${valueFormatted}`;
                    }
                }
            },
        },
        aspectRatio: 1,
    };
});


const formatCorrelationValue = (value) => {
    if (value === null || value === undefined || isNaN(value)) return '-';
    return value.toFixed(2);
};

const getCorrelationCellStyle = (value) => {
    const bgColor = getColorForValue(value); // Handles null/NaN
    let textColor = '#000000'; // Default black

     if (value !== null && value !== undefined && !isNaN(value)) {
         const intensity = Math.abs(value);
         // Basic luminance check approximation
         const r = parseInt(bgColor.slice(4, bgColor.indexOf(',')), 10);
         const g = parseInt(bgColor.slice(bgColor.indexOf(',') + 1, bgColor.lastIndexOf(',')), 10);
         const b = parseInt(bgColor.slice(bgColor.lastIndexOf(',') + 1, -1), 10);
         const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
         if (luminance < 0.5) { // If background is dark
              textColor = '#ffffff';
         }
     } else {
         textColor = '#6c757d'; // Grey text for N/A
     }

     return { backgroundColor: bgColor, color: textColor, textAlign: 'center' };
};


// --- Lifecycle Hooks ---

onMounted(() => {
    console.log("DatasetDetailView Mounted. Dataset ID:", datasetId.value);
    if (datasetId.value) {
        fetchDatasetDetails(datasetId.value);
    } else {
        error.value = "No Dataset ID provided in the URL.";
        loadingDataset.value = false;
    }
    // Chart.js plugins should be registered globally (e.g., in main.js)
    // import { Chart } from 'chart.js';
    // import { BoxPlotController, BoxAndWiskers } from '@sgratzl/chartjs-chart-boxplot';
    // import { MatrixController, MatrixElement } from 'chartjs-chart-matrix';
    // Chart.register(BoxPlotController, BoxAndWiskers, MatrixController, MatrixElement);
});

onUnmounted(() => {
   console.log("DatasetDetailView Unmounted. Cleaning up polling.");
   stopPolling(); // Clean up interval timer
});

// Watch for route changes
watch(datasetId, (newId, oldId) => {
  if (newId && newId !== oldId) {
    console.log(`Dataset ID changed from ${oldId} to ${newId}. Refetching data.`);
    fetchDatasetDetails(newId); // Refetch all data for the new dataset ID
  }
});

// Watch for visualization data changes
watch(visualizationData, (newData, oldData) => {
    // Reset view modes when visualization data first loads or changes significantly
    if (newData && (!oldData || JSON.stringify(newData) !== JSON.stringify(oldData))) {
        boxplotViewMode.value = 'stats';
        correlationViewMode.value = 'table';
    }
});


</script>

<style scoped>
/* --- Existing Styles (Keep As Is or Adjust) --- */
.dataset-detail-view {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  background-color: #f4f7f9; /* Light grey background for the page */
}

/* Header */
.header-section { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #dee2e6; gap: 1rem; flex-wrap: wrap; }
.header-left .dataset-title { margin: 0 0 0.25rem 0; color: #343a40; font-size: 2rem; font-weight: 600; }
.header-left .creation-date { font-size: 0.9rem; color: #6c757d; }
.header-right { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }

/* Description */
.description-section { display: flex; flex-wrap: wrap; gap: 2rem; margin-bottom: 2rem; background-color: #ffffff; padding: 1.5rem; border-radius: 8px; align-items: flex-start; border: 1px solid #e9ecef; }
.image-container { max-width: 300px; width: 100%; flex-shrink: 0; text-align: center; }
.image-container :deep(img.p-image-preview), .image-container :deep(img) { display: block; max-width: 100%; height: auto; object-fit: contain; max-height: 300px; border-radius: 4px; border: 1px solid #e9ecef; margin: 0 auto; background-color: #f8f9fa; } /* Added background for images */
.description-content { flex-grow: 1; min-width: 250px; }
.description-content h2 { margin-top: 0; margin-bottom: 1rem; color: #495057; font-size: 1.5rem; }
.description-content p { line-height: 1.6; color: #495057; word-break: break-word; }

/* Visualization Section */
.visualization-section {
  margin-top: 2.5rem;
  padding: 1.5rem;
  background-color: #ffffff;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.visualization-section h2 { margin-top: 0; margin-bottom: 1.5rem; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.75rem; font-size: 1.5rem; }
.start-visualization { text-align: center; padding: 2.5rem 1rem; background-color: #f8f9fa; border-radius: 6px; border: 1px dashed #ced4da; }
.start-visualization p { margin-bottom: 1.5rem; color: #6c757d; font-size: 1.05rem; }
.start-visualization .p-button { padding: 0.8rem 1.5rem; }
.loading-container.visualization-loading,
.error-container.visualization-error { min-height: 200px; margin-top: 1rem; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #f8f9fa; border-radius: 6px; padding: 1rem; text-align: center; }
.loading-container.visualization-loading p { margin-top: 1rem; font-size: 1.1rem; color: #6c757d; }
.loading-container.visualization-loading small { margin-top: 0.5rem; font-size: 0.9rem; color: #adb5bd; }
.error-container.visualization-error .p-message { width: 80%; max-width: 600px; margin-bottom: 1rem; }

/* Visualization Tabs */
.visualization-tabs { margin-top: 1.5rem; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav) { background: #e9ecef; border-radius: 6px 6px 0 0; border-bottom: none; padding: 0.5rem; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li .p-tabview-nav-link) { border: none; background: transparent; margin: 0 0.25rem; border-radius: 4px; transition: background-color 0.2s; color: #495057; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li:not(.p-highlight) .p-tabview-nav-link:hover) { background: #dee2e6; color: #343a40; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link) { background: #ffffff; color: var(--primary-color); border-bottom: none; font-weight: 600; }
.visualization-tabs :deep(.p-tabview-panels) { padding: 1.5rem; background-color: #ffffff; border: 1px solid #e9ecef; border-top: none; border-radius: 0 0 6px 6px; }

/* Chart Grid & Cards */
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
.charts-grid.stats-view { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.chart-card { border: 1px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: box-shadow 0.2s; border-radius: 6px; display: flex; flex-direction: column; background-color: #fff; }
.chart-card:hover { box-shadow: 0 3px 6px rgba(0,0,0,0.08); }
.chart-card :deep(.p-card-title) { font-size: 1.1rem; font-weight: 600; margin-bottom: 0; color: #495057; text-align: center; border-bottom: 1px solid #eee; padding: 0.75rem 1rem; background-color: #f8f9fa; border-radius: 6px 6px 0 0; }
.chart-card :deep(.p-card-content) { padding: 1rem; flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 300px; }
.single-chart-card :deep(.p-card-content) { min-height: 450px; padding: 1.5rem; }
.chart-card :deep(.p-card-content .p-chart) { width: 100%; height: 100%; }

/* Boxplot Stats View */
.data-card-simple :deep(.p-card-content) { min-height: auto; padding: 1rem; }
.stats-list { list-style: none; padding: 0; margin: 0; font-size: 0.95rem; color: #495057; }
.stats-list li { padding: 0.5rem 0; border-bottom: 1px dashed #eee; display: flex; justify-content: space-between; align-items: center; }
.stats-list li:last-child { border-bottom: none; }
.stat-label { color: #6c757d; margin-right: 1rem; }
.stat-value { font-weight: 500; color: #343a40; }

/* View Toggles */
.view-toggle-container { display: flex; justify-content: flex-end; margin-bottom: 1.5rem; }

/* Correlation Table */
.correlation-table-container { overflow-x: auto; } /* Enable horizontal scroll */
.correlation-table :deep(td),
.correlation-table :deep(th) { text-align: center !important; padding: 0.6rem 0.4rem !important; font-size: 0.9rem; white-space: nowrap; border: 1px solid #f1f3f5; } /* Lighter borders */
.correlation-table :deep(th) { background-color: #f8f9fa; font-weight: 600; color: #495057; position: sticky; top: 0; z-index: 1; }
.correlation-table :deep(td span) { display: inline-block; padding: 0.3rem 0.5rem; border-radius: 4px; min-width: 50px; font-weight: 500; line-height: 1.2; }
.correlation-table :deep(td:first-child),
.correlation-table :deep(th:first-child) { text-align: left !important; font-weight: bold; position: sticky; left: 0; z-index: 2; border-right: 2px solid #dee2e6 !important; background-color: #f8f9fa !important; }
.correlation-table :deep(th:first-child) { z-index: 3; } /* Header corner on top */

/* Main Tabs (Data Card, Training, Running) */
.main-tabs { margin-top: 2.5rem; }
.main-tabs :deep(.p-tabview-nav) { background: #ffffff; border-bottom: 1px solid #dee2e6; border-radius: 6px 6px 0 0; }
.main-tabs :deep(.p-tabview-nav .p-tabview-nav-link) { color: #495057; border-color: transparent; background: transparent;}
.main-tabs :deep(.p-tabview-nav .p-tabview-nav-link:not(.p-highlight):not(.p-disabled):hover) { background: #f8f9fa; border-color: transparent; color: #343a40;}
.main-tabs :deep(.p-tabview-nav .p-highlight .p-tabview-nav-link) { background: #ffffff; border-color: var(--primary-color); color: var(--primary-color); border-bottom: 2px solid var(--primary-color); } /* Standard underline style */
.main-tabs :deep(.p-tabview .p-tabview-panels) { padding: 1.5rem; background-color: #ffffff; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 6px 6px; }

/* Data Card Content */
.data-card-content { display: flex; flex-direction: column; gap: 2rem; }
.data-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem 1.5rem; }
.info-item { background-color: #f8f9fa; padding: 0.8rem 1rem; border-radius: 4px; border: 1px solid #e9ecef; }
.info-item label { display: block; font-weight: 600; color: #495057; margin-bottom: 0.4rem; font-size: 0.85rem; text-transform: uppercase; }
.info-item span, .info-item .p-tag { font-size: 0.95rem; color: #343a40; display: block; overflow: hidden; text-overflow: ellipsis; }
.info-item .p-tag { vertical-align: middle; }
.data-card-content h3 { margin-bottom: 1rem; margin-top: 1rem; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; font-size: 1.2rem; }
.preview-table-detail :deep(.p-datatable-tbody > tr > td) { padding: 0.5rem 0.8rem !important; font-size: 0.9rem; }
.preview-table-detail :deep(.p-datatable-thead > tr > th) { padding: 0.6rem 0.8rem !important; font-size: 0.9rem; background-color: #e9ecef; }

/* Training Results & Running Services Tabs */
.metrics-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.metric-chip { background-color: #e0e0e0; color: #333; padding: 0.3rem 0.7rem; border-radius: 16px; font-size: 0.85rem; white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
.main-tabs :deep(.p-tabview-panels .p-datatable td) { /* Ensure vertical align for buttons too */
    vertical-align: middle;
}
.main-tabs :deep(.p-tabview-panels code) { /* Style for API endpoint in Running Services */
   background-color: #e9ecef;
   padding: 0.2rem 0.4rem;
   border-radius: 3px;
   font-size: 0.85em;
   color: #495057;
   word-break: break-all; /* Break long model IDs if needed */
}

/* No Data / Error Messages in Tabs */
.main-tabs :deep(.p-tabview-panels .no-data-message),
.main-tabs :deep(.p-tabview-panels .error-message),
.visualization-tabs :deep(.p-tabview-panels .no-data-message)
 {
    padding: 2rem;
    text-align: center;
    color: #6c757d;
    background-color: #f8f9fa;
    border-radius: 4px;
    border: 1px dashed #dee2e6;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 150px;
    margin-top: 1rem; /* Add some space */
}
.main-tabs :deep(.p-tabview-panels .no-data-message i),
.visualization-tabs :deep(.p-tabview-panels .no-data-message i) {
    color: #ced4da;
}
.main-tabs :deep(.p-tabview-panels .error-message) {
    color: var(--red-700);
    background-color: var(--red-100);
    border: 1px solid var(--red-200);
}

/* Model Details Dialog */
.model-details-dialog { font-size: 0.95rem; }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem 1.5rem; margin-bottom: 1.5rem; background-color: #f8f9fa; padding: 1.25rem; border-radius: 6px; border: 1px solid #e9ecef; }
.detail-item label { font-weight: 600; color: #343a40; margin-right: 0.5rem; }
.params-section { margin-top: 1.5rem; margin-bottom: 1.5rem; }
.params-section h3 { margin-bottom: 0.75rem; color: #495057; }
.params-section pre { background-color: #e9ecef; padding: 1rem; border-radius: 4px; max-height: 250px; overflow-y: auto; font-size: 0.85rem; border: 1px solid #dee2e6; white-space: pre-wrap; word-break: break-all;}
.metrics-grid-dialog { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-top: 1rem; }
.metric-card-dialog { background: #ffffff; padding: 1rem; border-radius: 6px; text-align: center; border: 1px solid #dee2e6; transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
.metric-card-dialog:hover { transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.metric-title-dialog { font-weight: 600; color: #495057; margin-bottom: 0.5rem; font-size: 0.9rem; word-break: break-word; }
.metric-value-dialog { font-size: 1.3rem; color: #007bff; font-weight: 500; word-break: break-all;}

/* Main Loading / Error States */
.loading-container.main-loading, .error-container.main-error { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 400px; text-align: center; }
.loading-container p { margin-top: 1rem; font-size: 1.1rem; color: #6c757d; }
.error-container.main-error .p-message { margin-bottom: 1.5rem; }


/* Responsive Adjustments */
@media (max-width: 768px) {
  .dataset-detail-view { padding: 1rem; }
  .header-section { flex-direction: column; align-items: flex-start; gap: 1rem; }
  .header-right { width: 100%; justify-content: flex-start; }
  .description-section { flex-direction: column; }
  .image-container { max-width: 100%; max-height: 250px; }
  .image-container :deep(img) { max-height: 200px; }
  .charts-grid { grid-template-columns: 1fr; /* Stack cards */ }
  .correlation-table { font-size: 0.8rem; }
  .correlation-table :deep(td), .correlation-table :deep(th) { padding: 0.4rem 0.2rem !important; }
  .view-toggle-container { justify-content: center; /* Center toggle */ }
  .single-chart-card :deep(.p-card-content) { min-height: 350px; }
  .detail-grid { grid-template-columns: 1fr; } /* Stack details */
  .metrics-grid-dialog { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
}

</style>