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

    <!-- ======================== -->
    <!-- === NEW Markdown Info Section === -->
    <!-- ======================== -->
    <div class="markdown-info-section">
      <h2>Dataset Information & Usage</h2>

      <!-- Loader -->
      <div v-if="loadingMarkdown" class="loading-container markdown-loading">
        <ProgressSpinner style="width:30px;height:30px" strokeWidth="6" />
        <p>Loading info...</p>
      </div>

      <!-- Error -->
      <div v-else-if="markdownError" class="error-container markdown-error">
        <Message severity="warn" :closable="false">{{ markdownError }}</Message>
        <!-- Retry Button (if not a 404 error leading to generate button) -->
        <Button
            v-if="!showGenerateButton"
            label="Retry Loading Info"
            icon="pi pi-refresh"
            @click="fetchMarkdownInfo(datasetId)"
            class="p-button-sm p-button-text p-button-warning"
            style="margin-top: 0.5rem;"
        />
      </div>

      <!-- Generate Button -->
      <div v-else-if="showGenerateButton && !markdownContent" class="generate-button-container">
        <p>An information file with usage examples can be generated for this dataset.</p>
        <Button
          label="Generate Info File"
          icon="pi pi-file-edit"
          @click="generateMarkdownInfo"
          :loading="loadingMarkdown"
          class="p-button-raised p-button-info"
        />
      </div>

      <!-- Markdown Display -->
      <div v-else-if="markdownContent" class="markdown-content">
        <!-- Using vue3-markdown-it -->
        <VueMarkdownIt :source="markdownContent" />
        <!-- OR Using marked (example): -->
        <!-- <div ref="markedRenderer" v-html="renderedMarkdown"></div> -->
      </div>

      <!-- Fallback Message (rare case) -->
      <div v-else-if="!loadingMarkdown && !markdownError && !showGenerateButton && !markdownContent" class="no-data-message">
          <p>No information file available.</p>
      </div>
    </div>
    <!-- ======================== -->
    <!-- === END Markdown Info Section === -->
    <!-- ======================== -->

    <!-- Visualization Section -->
    <div class="visualization-section">
      <h2>Data Visualization</h2>

      <!-- Start Visualization Button -->
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

      <!-- Loading Indicator -->
      <div v-if="isLoadingVisualizations" class="loading-container visualization-loading">
        <ProgressSpinner style="width:50px;height:50px" strokeWidth="8" />
        <p>Generating visualizations... This may take a moment.</p>
        <small v-if="currentVisualizationTaskId">(Task ID: {{ currentVisualizationTaskId }})</small>
      </div>

      <!-- Error Message -->
      <div v-if="visualizationError && !isLoadingVisualizations" class="error-container visualization-error">
        <Message severity="error" :closable="false">
          Failed to generate visualizations: {{ visualizationError }}
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

      <!-- Visualization Tabs -->
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
                 <Card class="chart-card single-chart-card"> <!-- Heatmap still makes sense as a single card -->
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

    <!-- Main Tabs (Data Card / Training Results) -->
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
                <Column header="Metrics"> <template #body="{data}"> <div class="metrics-list"> <span v-for="(value, key) in data.metrics" :key="key" class="metric-chip"> {{ key }}: {{ value.toFixed ? value.toFixed(3) : value }} </span> </div> </template> </Column>
                <Column field="start_time" header="Trained On" :sortable="true"> <template #body="{data}"> {{ formatDateTime(data.start_time) }} </template> </Column>
                <Column header="Actions"> <template #body="{data}"> <Button icon="pi pi-chart-line" class="p-button-sm p-button-info p-button-text" @click="showModelDetails(data)" v-tooltip.top="'View Training Details'" /> <Button icon="pi pi-play" class="p-button-sm p-button-success p-button-text" @click="startInference(data.id)" v-tooltip.top="'Start Inference Service'" :disabled="isModelRunning(data.id)" /> </template> </Column>
             </DataTable>
         </div>
        </TabPanel>
      <!-- === NEW Inference Models Tab === -->
      <TabPanel header="Inference Models">
        <div class="inference-models-content">
          <!-- Сообщение, если нет запущенных моделей для этого датасета -->
          <div v-if="!runningInferenceModelsForDataset || runningInferenceModelsForDataset.length === 0" class="no-data-message">
              <i class="pi pi-power-off" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></i>
              <p>No active inference services found for this dataset.</p>
              <small>You can start one from the 'Training Results' tab.</small>
          </div>

          <!-- Таблица с запущенными моделями -->
          <DataTable v-else :value="runningInferenceModelsForDataset" responsiveLayout="scroll" class="p-datatable-striped p-datatable-sm">
              <Column field="model_type" header="Model Type" :sortable="true"></Column>
              <Column field="target_column" header="Target" :sortable="true">
                  <template #body="{data}">
                      <Tag :value="data.target_column || 'N/A'" />
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
               <Column field="status" header="Status" :sortable="true">
                  <template #body="{data}">
                      <Tag :value="data.status" :severity="data.status === 'running' ? 'success' : 'warning'" />
                  </template>
              </Column>
               <!-- Опционально: Показать API URL (может быть полезно для разработчиков) -->
               <Column header="API">
        <template #body="{data}">
            <Button
                icon="pi pi-book"
                class="p-button-sm p-button-secondary"
                @click="openApiUrl(data.api_url)"
            />
        </template>
    </Column>
               <Column header="Actions">
                  <template #body="{data}">
                      <Button
                          icon="pi pi-stop-circle"
                          class="p-button-sm p-button-danger p-button-text"
                          @click="stopInference(data.model_id)"
                          v-tooltip.top="'Stop Inference Service'"
                          :disabled="data.status !== 'running'"
                       />
                       <!-- Можно добавить кнопку для перехода к документации API инференса -->
                  </template>
              </Column>
          </DataTable>
        </div>
      </TabPanel>    
    
      </TabView>
    
    <!-- Model Details Dialog -->
    <Dialog v-model:visible="showDetailsDialog" header="Model Training Details" :modal="true" :style="{ width: '60vw', minWidth: '400px' }" :breakpoints="{'960px': '75vw', '640px': '90vw'}" dismissableMask >
      <div v-if="selectedModel" class="model-details-dialog">
        <div class="detail-grid">
            <div class="detail-item"><label>Model ID:</label> <span>{{ selectedModel.id }}</span></div>
            <div class="detail-item"><label>Model Type:</label> <span>{{ selectedModel.model_type }}</span></div>
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
            <div class="metric-title-dialog">{{ key }}</div> <div class="metric-value-dialog">{{ value.toFixed ? value.toFixed(4) : value }}</div>
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

// --- NEW Markdown Renderer ---
import VueMarkdownIt from 'vue3-markdown-it';
// If using marked:
// import { marked } from 'marked';
// const markedRenderer = ref(null);

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
import Tooltip from 'primevue/tooltip';
import SelectButton from 'primevue/selectbutton';

// Directives need to be registered if not done globally
const vTooltip = Tooltip;

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'; // Use env var
const route = useRoute();
const router = useRouter();
const toast = useToast();

// --- State ---
const dataset = ref(null);
const trainingResults = ref([]);
const filePreview = ref([]);
const columnInfo = ref([]);
const runningModels = ref([]);
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

// --- NEW State for Markdown ---
const markdownContent = ref(null);
const loadingMarkdown = ref(false);
const markdownError = ref(null);
const showGenerateButton = ref(false); // Controls visibility of the generate button

// --- State for Visualization ---
const visualizationData = ref(null);
const isLoadingVisualizations = ref(false);
const isStartingVisualization = ref(false);
const visualizationError = ref(null);
const activeVisualizationTabIndex = ref(0);
const pollingIntervalId = ref(null);
const currentVisualizationTaskId = ref(null);

// --- State for View Toggles ---
const boxplotViewMode = ref('stats');
const boxplotViewOptions = ref([
    {label: 'Statistics', value: 'stats'},
    {label: 'Chart', value: 'chart'}
]);
const correlationViewMode = ref('table');
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
  // { separator: true },
  // { label: 'Delete Dataset', icon: 'pi pi-trash', command: () => deleteCurrentDataset() }
]);

// --- Methods ---

const runningInferenceModelsForDataset = computed(() => {
  if (!dataset.value || !runningModels.value) {
    return [];
  }
  // Фильтруем массив runningModels по имени файла текущего датасета
  return runningModels.value.filter(model => model.dataset_filename === dataset.value.filename);
});

const goToTrain = (datasetId)=>{
  router.push({ path: '/', query: { dataset: datasetId } });
};

// --- NEW Markdown Methods ---
const fetchMarkdownInfo = async (id) => {
  if (!id) return;
  console.log("Fetching Markdown info for dataset:", id);
  loadingMarkdown.value = true;
  markdownError.value = null;
  markdownContent.value = null; // Reset before fetching
  showGenerateButton.value = false; // Reset button visibility

  try {
    // Assuming the backend returns plain text markdown
    const response = await axios.get(`${API_BASE_URL}/dataset/${id}/info_markdown`, {
       // Ensure axios doesn't try to parse it as JSON if backend sends text/plain
       // Usually axios handles text/markdown correctly, but this can help if needed:
       // transformResponse: [(data) => data] // Keep response as string
    });
    markdownContent.value = response.data; // Data should be the markdown string
    console.log("Markdown content fetched successfully.");
  } catch (err) {
    if (err.response && err.response.status === 404) {
      console.log("Info Markdown file not found (404). Showing generate button.");
      markdownError.value = null; // 404 is not an app error, but "file not found" state
      showGenerateButton.value = true; // Show the button to generate
    } else {
      console.error('Error fetching markdown info:', err);
      markdownError.value = `Failed to load info file: ${err.response?.data?.detail || err.message}`;
      showGenerateButton.value = false; // Don't show generate button on other errors
    }
    markdownContent.value = null; // Clear content on any error
  } finally {
    loadingMarkdown.value = false;
  }
};

const generateMarkdownInfo = async () => {
  if (!datasetId.value) return;
  console.log("Requesting Markdown generation for dataset:", datasetId.value);
  loadingMarkdown.value = true; // Show loader during generation request + fetch
  markdownError.value = null;
  showGenerateButton.value = false; // Hide button while processing

  try {
    await axios.post(`${API_BASE_URL}/dataset/${datasetId.value}/generate_info_markdown`);
    toast.add({ severity: 'success', summary: 'Success', detail: 'Info file generated. Fetching content...', life: 3000 });
    // After successful generation request, immediately try to fetch the content
    await fetchMarkdownInfo(datasetId.value);
  } catch (err) {
    console.error('Error generating markdown info:', err);
    markdownError.value = `Failed to generate info file: ${err.response?.data?.detail || err.message}`;
    toast.add({ severity: 'error', summary: 'Generation Failed', detail: markdownError.value, life: 5000 });
    showGenerateButton.value = true; // Show button again on failure to allow retry
    loadingMarkdown.value = false; // Stop loading indicator on generation error
  }
  // No finally block needed here, fetchMarkdownInfo will set loadingMarkdown to false
};


// --- MODIFIED fetchDatasetDetails ---
const fetchDatasetDetails = async (id) => {
  loadingDataset.value = true;
  error.value = null;
  dataset.value = null;
  filePreview.value = [];
  columnInfo.value = [];
  trainingResults.value = [];
  visualizationData.value = null;
  visualizationError.value = null;
  isLoadingVisualizations.value = false;
  stopPolling();

  // --- Reset Markdown State ---
  markdownContent.value = null;
  loadingMarkdown.value = false;
  markdownError.value = null;
  showGenerateButton.value = false;
  // -----------------------------

  try {
    const response = await axios.get(`${API_BASE_URL}/dataset/${id}`);
    dataset.value = response.data;

    // Process column types if available
    if (dataset.value.column_types) {
        columnInfo.value = Object.entries(dataset.value.column_types).map(([name, type]) => ({ name, type }));
    } else if (dataset.value.columns) {
        columnInfo.value = dataset.value.columns.map(name => ({ name, type: 'Unknown' }));
    }

    // Fetch related data only if dataset loaded successfully
    if (dataset.value?.filename) {
      // Fetch these concurrently
      await Promise.all([
        fetchTrainingResults(dataset.value.filename),
        fetchDatasetPreview(dataset.value.filename),
        fetchRunningModels(),
        fetchVisualizations(),
        fetchMarkdownInfo(dataset.value.id) // <<<--- Fetch Markdown Info Here
      ]);
    } else if (!dataset.value) {
        // Handle case where API returned success but no data
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

const fetchRunningModels = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/running_models/`);
        runningModels.value = response.data;
    } catch (error) {
        console.error('Failed to fetch running models status:', error);
        // toast.add({ severity: 'warn', summary: 'Network Issue', detail: 'Could not fetch running model status.', life: 2000 });
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
    link.setAttribute('download', filename.split('_').slice(1).join('_') || filename);
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
    toast.add({ severity: 'warn', summary: 'Warning', detail: 'Could not copy link automatically. You can copy it from the address bar.', life: 5000 });
  });
};

const toggleShareMenu = (event) => {
  shareMenu.value.toggle(event);
};

const startInference = async (modelId) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/start_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Inference Started',
      detail: `Model ${modelId} inference service started. Status: ${response.data.status}`,
      life: 4000
    });
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

const stopInference = async (modelId) => {
  if (!modelId) return;
  toast.add({ severity: 'info', summary: 'Stopping...', detail: `Requesting to stop inference for model ${modelId}`, life: 2000 });

  try {
    await axios.delete(`${API_BASE_URL}/stop_inference/${modelId}`);
    toast.add({
      severity: 'success',
      summary: 'Success',
      detail: `Inference service for model ${modelId} stopped successfully.`,
      life: 4000
    });
    // Обновляем список запущенных моделей после успешной остановки
    // Небольшая задержка, чтобы дать бэкенду время обработать
    setTimeout(fetchRunningModels, 500);
  } catch (error) {
    console.error(`Error stopping inference for model ${modelId}:`, error);
    toast.add({
      severity: 'error',
      summary: 'Error Stopping Inference',
      detail: `Failed to stop inference for ${modelId}: ${error.response?.data?.detail || error.message}`,
      life: 5000
    });
  }
};

const openApiUrl = (api) => {
    if (!api) return;
    window.open(api, '_blank'); // Open in new tab
};

const showModelDetails = (model) => {
  selectedModel.value = model;
  showDetailsDialog.value = true;
};

const goBack = () => {
    router.push({ name: 'Datasets' }); // Assuming a route named 'Datasets' exists
}

const isModelRunning = (modelId) => {
    return runningModels.value.some(m => m.model_id === modelId && m.status === 'running');
};

// --- Visualization Methods ---

const fetchVisualizations = async () => {
  if (!datasetId.value) return;
  visualizationError.value = null;
  visualizationData.value = null;

  try {
    const response = await axios.get(`${API_BASE_URL}/datasets/${datasetId.value}/visualizations`);
    if (response.data) {
        const vizResult = response.data;
        if (vizResult.status === 'SUCCESS') {
            visualizationData.value = vizResult.visualization_data;
            boxplotViewMode.value = 'stats';
            correlationViewMode.value = 'table';
        } else if (vizResult.status === 'PENDING') {
             console.warn("Visualization task is currently pending.");
             isLoadingVisualizations.value = true;
             // Optional: Start polling if backend provides task_id here
             // currentVisualizationTaskId.value = vizResult.task_id;
             // if(currentVisualizationTaskId.value) startPolling(currentVisualizationTaskId.value);
        } else if (vizResult.status === 'FAILURE') {
            visualizationError.value = vizResult.error_message || 'Unknown error during previous visualization.';
        } else {
             visualizationData.value = null; // Treat other statuses as not ready
        }
    } else {
         visualizationData.value = null; // No visualization generated yet
    }
  } catch (error) {
    if (error.response && error.response.status === 404) {
         visualizationData.value = null; // No record found
    } else {
        console.error('Error fetching visualization data:', error);
        visualizationError.value = `Could not load existing visualizations: ${error.message}`;
    }
  }
  // Loading state is mainly controlled by the overall fetch or polling
};


const startVisualization = async () => {
  if (!datasetId.value) return;
  isStartingVisualization.value = true;
  isLoadingVisualizations.value = true;
  visualizationError.value = null;
  visualizationData.value = null;
  stopPolling();

  try {
    const response = await axios.post(`${API_BASE_URL}/datasets/${datasetId.value}/visualize`);
    currentVisualizationTaskId.value = response.data.task_id;
    toast.add({ severity: 'info', summary: 'Processing', detail: 'Visualization generation started...', life: 3000 });
    startPolling(currentVisualizationTaskId.value);
  } catch (error) {
    console.error('Error starting visualization task:', error);
    visualizationError.value = `Failed to start visualization: ${error.response?.data?.detail || error.message}`;
    isLoadingVisualizations.value = false;
    toast.add({ severity: 'error', summary: 'Error', detail: visualizationError.value, life: 5000 });
  } finally {
    isStartingVisualization.value = false;
  }
};

const pollVisualizationStatus = async (taskId) => {
  if (!taskId || taskId !== currentVisualizationTaskId.value || !pollingIntervalId.value) {
      return; // Stop if polling was cancelled or task ID changed
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/visualization_status/${taskId}`);
    const { status, error } = response.data; // Use 'error' which contains detailed info on failure

    if (status === 'SUCCESS') {
      stopPolling();
      toast.add({ severity: 'success', summary: 'Success', detail: 'Visualizations generated!', life: 3000 });
      isLoadingVisualizations.value = false;
      await fetchVisualizations(); // Fetch the final data

    } else if (status === 'FAILURE') {
      stopPolling();
      // Use detailed error message if available from backend structure
      const errorMessage = error?.message || error?.error_message || 'Unknown error during visualization task.';
      console.error('Visualization task failed:', error); // Log the full error object
      visualizationError.value = errorMessage;
      toast.add({ severity: 'error', summary: 'Visualization Failed', detail: errorMessage, life: 6000 });
      isLoadingVisualizations.value = false;

    } else if (status === 'PENDING' || status === 'STARTED' || status === 'RETRY') {
      isLoadingVisualizations.value = true; // Keep loader active
    } else {
         console.warn(`Unknown task status received: ${status}. Stopping poll.`);
         stopPolling();
         visualizationError.value = `Received an unexpected status: ${status}`;
         isLoadingVisualizations.value = false;
    }
  } catch (err) {
    console.error('Error polling visualization status:', err);
    if (err.response && err.response.status === 404) {
        console.error(`Task ${taskId} not found during polling. Stopping poll.`);
        stopPolling();
        visualizationError.value = `Visualization task ${taskId} could not be found.`;
        isLoadingVisualizations.value = false;
    }
    // Let it retry on the next interval for transient network issues otherwise
  }
};


const startPolling = (taskId) => {
  if (!taskId) return;
  stopPolling();
  currentVisualizationTaskId.value = taskId;
  isLoadingVisualizations.value = true;
  setTimeout(() => pollVisualizationStatus(taskId), 1000); // Initial check
  pollingIntervalId.value = setInterval(() => {
    pollVisualizationStatus(taskId);
  }, 5000); // Poll every 5 seconds
};

const stopPolling = () => {
  if (pollingIntervalId.value) {
    clearInterval(pollingIntervalId.value);
    pollingIntervalId.value = null;
  }
  // Don't set isLoadingVisualizations here; let poll status handle it.
};


// --- Computed & Watchers ---

const getImageUrl = (ds) => {
    if (ds?.imageUrl) {
        if (ds.imageUrl.startsWith('http://') || ds.imageUrl.startsWith('https://')) {
            return ds.imageUrl;
        }
        if (ds.imageUrl.startsWith('/')) {
             return `${API_BASE_URL}${ds.imageUrl}`;
        }
         // Assume it's a relative path from API base if it doesn't start with '/' or 'http'
         // This handles the case where backend returns "/dataset/{id}/image"
         return `${API_BASE_URL}${ds.imageUrl.startsWith('/') ? '' : '/'}${ds.imageUrl}`;
    }
    // Fallback placeholder URL structure needs to match backend endpoint
    return `${API_BASE_URL}/placeholder.png`; // Assuming backend serves this at the root
};

const formattedDate = (dateString) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric', month: 'long', day: 'numeric'
  });
};

const formatDateTime = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString(undefined, {
        year: 'numeric', month: 'numeric', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });
};

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  if (!bytes || bytes < 0) return 'N/A';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const index = Math.min(i, sizes.length - 1);
  return parseFloat((bytes / Math.pow(k, index)).toFixed(2)) + ' ' + sizes[index];
};

const formatFilename = (filename) => {
    return filename?.replace(/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_/, '') || filename || 'N/A';
};

// --- Chart Computed Properties ---

const preparedHistograms = computed(() => {
  if (!visualizationData.value?.histograms) return [];
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
        datasets: [{
            label: 'Frequency', data: hist.counts,
            backgroundColor: '#A0C4FF', borderColor: '#4A90E2', borderWidth: 1,
            barPercentage: 1.0, categoryPercentage: 1.0,
        }]
      },
      chartOptions: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Frequency' } },
          x: { title: { display: true, text: 'Value Bins' }, ticks: { maxRotation: 45, minRotation: 0 } }
        },
        plugins: {
            legend: { display: false },
            tooltip: { callbacks: {
                    title: (tooltipItems) => `Bin: ${tooltipItems[0].label}`,
                    label: (context) => ` Frequency: ${context.parsed.y}`
            }}
        }
      }
    };
  });
});

const preparedIndividualBoxplots = computed(() => {
    const boxplotData = visualizationData.value?.boxplots;
    if (!boxplotData || boxplotData.length === 0) return [];
    const defaultChartOptions = {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { beginAtZero: false, title: { display: true, text: 'Value' } }, x: { title: { display: false } } },
        plugins: {
            legend: { display: false }, title: { display: false },
            tooltip: { callbacks: {
                    label: function(context) {
                        if (context.datasetIndex == null || context.dataIndex == null) return '';
                        const item = context.chart.data.datasets[context.datasetIndex]?.data[context.dataIndex];
                        if (!item) return '';
                        return [`Max: ${item.max?.toFixed(2) ?? 'N/A'}`, `Q3: ${item.q3?.toFixed(2) ?? 'N/A'}`, `Median: ${item.median?.toFixed(2) ?? 'N/A'}`, `Q1: ${item.q1?.toFixed(2) ?? 'N/A'}`, `Min: ${item.min?.toFixed(2) ?? 'N/A'}`];
                    }
            }}
        }
    };
    return boxplotData.map(bp => ({
        column: bp.column,
        chartData: {
            labels: [bp.column],
            datasets: [{
                label: bp.column,
                data: [{ min: bp.min ?? undefined, q1: bp.q1 ?? undefined, median: bp.median ?? undefined, q3: bp.q3 ?? undefined, max: bp.max ?? undefined }],
                backgroundColor: 'rgba(160, 196, 255, 0.5)', borderColor: '#4A90E2', borderWidth: 1,
                itemRadius: 3, itemStyle: 'circle', outlierStyle: 'circle'
            }]
        },
        chartOptions: defaultChartOptions
    }));
});

const preparedCorrelationData = computed(() => {
  const matrix = visualizationData.value?.correlation_matrix;
  if (!matrix || !matrix.columns || !matrix.data) return [];
  return matrix.data.map((row, rowIndex) => {
    if (rowIndex >= matrix.columns.length) return null;
    const rowData = { column: matrix.columns[rowIndex] };
    row.forEach((value, colIndex) => {
       if (colIndex < matrix.columns.length) { rowData[matrix.columns[colIndex]] = value; }
    });
    return rowData;
  }).filter(row => row !== null);
});

const preparedCorrelationMatrixChartData = computed(() => {
  const matrix = visualizationData.value?.correlation_matrix;
  if (!matrix || !matrix.columns || !matrix.data || matrix.columns.length < 2) return { datasets: [] };
  const labels = matrix.columns;
  const data = [];
  matrix.data.forEach((row, i) => {
    const yLabel = labels[i];
    row.forEach((value, j) => {
       const xLabel = labels[j];
       if (value !== null && value !== undefined && !isNaN(value)) {
         data.push({ x: xLabel, y: yLabel, v: value });
       }
    });
  });
  return {
      datasets: [{
          label: 'Correlation', data: data,
          backgroundColor: function(context) {
             const value = context.dataset.data[context.dataIndex]?.v;
             return getColorForValue(value);
          },
          borderColor: 'rgba(0,0,0,0.1)', borderWidth: 1,
          width: ({chart}) => (chart.chartArea?.width ?? 300) / labels.length - 1,
          height: ({chart}) => (chart.chartArea?.height ?? 300) / labels.length - 1,
      }]
  };
});

const getColorForValue = (value) => {
    if (value === null || value === undefined || isNaN(value)) return 'rgba(200, 200, 200, 0.5)';
    const whitePoint = 255;
    let r, g, b;
    if (value > 0) { r = Math.round(whitePoint * (1 - value)); g = Math.round(whitePoint * (1 - value)); b = whitePoint; }
    else { r = whitePoint; g = Math.round(whitePoint * (1 + value)); b = Math.round(whitePoint * (1 + value)); }
    r = Math.max(0, Math.min(255, r)); g = Math.max(0, Math.min(255, g)); b = Math.max(0, Math.min(255, b));
    return `rgb(${r}, ${g}, ${b})`;
};

const correlationMatrixChartOptions = computed(() => {
   const labels = visualizationData.value?.correlation_matrix?.columns ?? [];
   return {
        responsive: true, maintainAspectRatio: false, aspectRatio: 1,
        scales: {
            x: { type: 'category', labels: labels, position: 'bottom', ticks: { display: true, autoSkip: false, maxRotation: 90, minRotation: 45 }, grid: { display: false } },
            y: { type: 'category', labels: labels, position: 'left', offset: true, ticks: { display: true, autoSkip: false }, grid: { display: false }, reverse: true }
        },
        plugins: {
            legend: { display: false },
            tooltip: { callbacks: { title: function() { return ''; }, label: function(context) { const item = context.dataset.data[context.dataIndex]; return item ? `Corr(${item.y}, ${item.x}): ${item.v.toFixed(3)}` : ''; } } }
        }
    };
});

const formatCorrelationValue = (value) => {
    if (value === null || value === undefined || isNaN(value)) return '-';
    return value.toFixed(2);
};

const getCorrelationCellStyle = (value) => {
    const bgColor = getColorForValue(value);
    let textColor = '#000000';
    if (value !== null && value !== undefined && !isNaN(value)) {
         if (Math.abs(value) > 0.5) { textColor = '#ffffff'; }
    } else { textColor = '#6c757d'; }
    return { backgroundColor: bgColor, color: textColor };
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
    // Ensure Chart.js plugins (BoxPlot, Matrix) are registered globally (e.g., in main.js)
});

onUnmounted(() => {
   console.log("DatasetDetailView Unmounted. Cleaning up polling.");
   stopPolling(); // Clean up interval timer
});

// Watch for route changes if navigating between detail pages
watch(datasetId, (newId, oldId) => {
  if (newId && newId !== oldId) {
    console.log(`Dataset ID changed from ${oldId} to ${newId}. Refetching data.`);
    fetchDatasetDetails(newId);
  }
});

// Optional: Watch for visualization data changes if needed
watch(visualizationData, (newData, oldData) => {
    if (newData && !oldData) { // When data first loads
        boxplotViewMode.value = 'stats';
        correlationViewMode.value = 'table';
    }
});

// If using marked:
// const renderedMarkdown = computed(() => {
//   if (markdownContent.value) {
//     return marked.parse(markdownContent.value);
//   }
//   return '';
// });

</script>

<style scoped>
/* --- Existing Styles (Keep As Is or Adjust as Needed) --- */
.dataset-detail-view { padding: 2rem; max-width: 1200px; margin: 0 auto; background-color: #f4f7f9; }
.header-section { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #dee2e6; }
.header-left .dataset-title { margin: 0 0 0.25rem 0; color: #343a40; font-size: 2rem; font-weight: 600; }
.header-left .creation-date { font-size: 0.9rem; color: #6c757d; }
.header-right { display: flex; gap: 0.75rem; align-items: center; }
.description-section { display: flex; flex-wrap: wrap; gap: 2rem; margin-bottom: 2rem; background-color: #ffffff; padding: 1.5rem; border-radius: 8px; align-items: flex-start; border: 1px solid #e9ecef; }
.image-container { max-width: 300px; width: 100%; flex-shrink: 0; text-align: center; }
.image-container :deep(img.p-image-preview), .image-container :deep(img) { display: block; max-width: 100%; height: auto; object-fit: contain; max-height: 300px; border-radius: 4px; border: 1px solid #e9ecef; margin: 0 auto; }
.description-content { flex-grow: 1; min-width: 250px; }
.description-content h2 { margin-top: 0; margin-bottom: 1rem; color: #495057; font-size: 1.5rem; }
.description-content p { line-height: 1.6; color: #495057; word-break: break-word; }

/* --- Visualization Section Styles --- */
.visualization-section { margin-top: 2.5rem; padding: 1.5rem; background-color: #ffffff; border-radius: 8px; border: 1px solid #e9ecef; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
.visualization-section h2 { margin-top: 0; margin-bottom: 1.5rem; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.75rem; font-size: 1.5rem; }
.start-visualization { text-align: center; padding: 2.5rem 1rem; background-color: #f8f9fa; border-radius: 6px; border: 1px dashed #ced4da; }
.start-visualization p { margin-bottom: 1.5rem; color: #6c757d; font-size: 1.05rem; }
.start-visualization .p-button { padding: 0.8rem 1.5rem; }
.loading-container.visualization-loading, .error-container.visualization-error { min-height: 200px; margin-top: 1rem; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #f8f9fa; border-radius: 6px; padding: 1rem; }
.loading-container.visualization-loading p { margin-top: 1rem; font-size: 1.1rem; color: #6c757d; }
.loading-container.visualization-loading small { margin-top: 0.5rem; font-size: 0.9rem; color: #adb5bd; }
.error-container.visualization-error .p-message { width: 80%; max-width: 600px; margin-bottom: 1rem; }
.visualization-tabs { margin-top: 1.5rem; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav) { background: #e9ecef; border-radius: 6px 6px 0 0; border-bottom: none; padding: 0.5rem; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li .p-tabview-nav-link) { border: none; background: transparent; margin: 0 0.25rem; border-radius: 4px; transition: background-color 0.2s; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li:not(.p-highlight) .p-tabview-nav-link:hover) { background: #dee2e6; }
.visualization-tabs.p-tabview-cards :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link) { background: #ffffff; color: var(--primary-color); }
.visualization-tabs :deep(.p-tabview-panels) { padding: 1.5rem; background-color: #ffffff; border: 1px solid #e9ecef; border-top: none; border-radius: 0 0 6px 6px; }
.charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
.charts-grid.stats-view { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.chart-card { border: 1px solid #e9ecef; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: box-shadow 0.2s; border-radius: 6px; display: flex; flex-direction: column; }
.chart-card:hover { box-shadow: 0 3px 6px rgba(0,0,0,0.08); }
.chart-card :deep(.p-card-title) { font-size: 1.1rem; font-weight: 600; margin-bottom: 0; color: #495057; text-align: center; border-bottom: 1px solid #eee; padding: 0.75rem 1rem; }
.chart-card :deep(.p-card-content) { padding: 1rem; flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 300px; }
.single-chart-card :deep(.p-card-content) { min-height: 450px; padding: 1.5rem; }
.chart-card :deep(.p-card-content .p-chart) { width: 100%; height: 100%; }
.data-card-simple :deep(.p-card-content) { min-height: auto; padding: 1rem; }
.stats-list { list-style: none; padding: 0; margin: 0; font-size: 0.95rem; color: #495057; }
.stats-list li { padding: 0.5rem 0; border-bottom: 1px dashed #eee; display: flex; justify-content: space-between; }
.stats-list li:last-child { border-bottom: none; }
.stat-label { color: #6c757d; margin-right: 1rem; }
.stat-value { font-weight: 500; color: #343a40; }
.no-data-message { padding: 2rem; text-align: center; color: #6c757d; background-color: #f8f9fa; border-radius: 4px; border: 1px dashed #dee2e6; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 150px; }
.no-data-message i { color: #ced4da; font-size: 1.5rem; margin-bottom: 0.5rem; }
.correlation-table :deep(td), .correlation-table :deep(th) { text-align: center !important; padding: 0.6rem 0.4rem !important; font-size: 0.9rem; white-space: nowrap; border: 1px solid #eee; }
.correlation-table :deep(th) { background-color: #f8f9fa; font-weight: 600; color: #495057; position: sticky; top: 0; z-index: 1; }
.correlation-table :deep(td span) { display: inline-block; padding: 0.3rem 0.5rem; border-radius: 4px; min-width: 50px; font-weight: 500; border: 1px solid rgba(0,0,0,0.05); line-height: 1.2; }
.correlation-table :deep(td:first-child), .correlation-table :deep(th:first-child) { text-align: left !important; font-weight: bold; position: sticky; left: 0; z-index: 2; border-right: 2px solid #dee2e6 !important; background-color: #f8f9fa !important; color: inherit; }
.correlation-table :deep(th:first-child) { z-index: 3; }
.view-toggle-container { display: flex; justify-content: flex-end; margin-bottom: 1.5rem; }
.chart-container { margin-top: 1rem; }
.boxplot-chart-container, .heatmap-chart-container { width: 100%; }

/* --- Main Tabs, Data Card, Training Results, Dialog Styles --- */
.main-tabs { margin-top: 2rem; }
.main-tabs :deep(.p-tabview-nav) { background: #ffffff; border-bottom: 1px solid #dee2e6; border-radius: 6px 6px 0 0; }
.main-tabs :deep(.p-tabview .p-tabview-panels) { padding: 1.5rem; background-color: #ffffff; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 6px 6px; }
.data-card-content { display: flex; flex-direction: column; gap: 2rem; }
.data-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem 1.5rem; }
.info-item { background-color: #f8f9fa; padding: 0.8rem 1rem; border-radius: 4px; border: 1px solid #e9ecef; }
.info-item label { display: block; font-weight: 600; color: #495057; margin-bottom: 0.4rem; font-size: 0.85rem; text-transform: uppercase; }
.info-item span, .info-item .p-tag { font-size: 0.95rem; color: #343a40; word-break: break-word; }
.info-item .p-tag { vertical-align: middle; }
.data-card-content h3 { margin-bottom: 1rem; margin-top: 1rem; color: #495057; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; font-size: 1.2rem; }
.preview-table-detail :deep(.p-datatable-tbody > tr > td) { padding: 0.5rem 0.8rem !important; font-size: 0.9rem; }
.preview-table-detail :deep(.p-datatable-thead > tr > th) { padding: 0.6rem 0.8rem !important; font-size: 0.9rem; background-color: #e9ecef; }
.metrics-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.metric-chip { background-color: #e0e0e0; color: #333; padding: 0.3rem 0.7rem; border-radius: 16px; font-size: 0.85rem; white-space: nowrap; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
.model-details-dialog { font-size: 0.95rem; }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem 1.5rem; margin-bottom: 1.5rem; background-color: #f8f9fa; padding: 1.25rem; border-radius: 6px; border: 1px solid #e9ecef; }
.detail-item label { font-weight: 600; color: #343a40; margin-right: 0.5rem; }
.params-section { margin-top: 1.5rem; margin-bottom: 1.5rem; }
.params-section h3 { margin-bottom: 0.75rem; color: #495057; }
.params-section pre { background-color: #e9ecef; padding: 1rem; border-radius: 4px; max-height: 250px; overflow-y: auto; font-size: 0.85rem; border: 1px solid #dee2e6; white-space: pre-wrap; word-break: break-all;}
.metrics-grid-dialog { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-top: 1rem; }
.metric-card-dialog { background: #ffffff; padding: 1rem; border-radius: 6px; text-align: center; border: 1px solid #dee2e6; transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
.metric-card-dialog:hover { transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.metric-title-dialog { font-weight: 600; color: #495057; margin-bottom: 0.5rem; font-size: 0.9rem; }
.metric-value-dialog { font-size: 1.3rem; color: #007bff; font-weight: 500; }
.loading-container.main-loading, .error-container.main-error { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 400px; text-align: center; }
.loading-container p { margin-top: 1rem; font-size: 1.1rem; color: #6c757d; }
.error-container .p-message { margin-bottom: 1.5rem; }
.error-message { color: var(--red-700); background-color: var(--red-100); padding: 1rem; border-radius: 4px; border: 1px solid var(--red-200); margin-top: 1rem; margin-bottom: 1rem; }
.code-inline { background-color: #e9ecef; padding: 0.1rem 0.3rem; border-radius: 3px; font-family: monospace; font-size: 0.9em; }

/* --- NEW Styles for Markdown Info Section --- */
.markdown-info-section {
  margin-top: 2.5rem;
  padding: 1.5rem;
  background-color: #ffffff;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.markdown-info-section h2 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  color: #495057;
  border-bottom: 1px solid #eee;
  padding-bottom: 0.75rem;
  font-size: 1.5rem;
}
.loading-container.markdown-loading,
.error-container.markdown-error,
.generate-button-container {
  min-height: 100px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  background-color: #f8f9fa;
  border-radius: 6px;
  padding: 1.5rem;
  margin-top: 1rem;
}
.loading-container.markdown-loading p {
   margin-top: 0.8rem;
   color: #6c757d;
   font-size: 1rem;
}
.error-container.markdown-error .p-message {
   width: 90%;
   max-width: 700px;
}
.generate-button-container p {
    margin-bottom: 1.2rem;
    color: #6c757d;
    font-size: 1.05rem;
}
.generate-button-container .p-button {
     padding: 0.7rem 1.3rem;
}
.markdown-content {
   margin-top: 1rem;
   line-height: 1.7;
   color: #343a40;
   max-width: 100%; /* Ensure content doesn't overflow container */
   overflow-x: hidden; /* Prevent horizontal scroll from content itself */
}

/* --- Deep Styling for Rendered Markdown Content --- */
.markdown-content :deep(h1),
.markdown-content :deep(h2),
.markdown-content :deep(h3),
.markdown-content :deep(h4),
.markdown-content :deep(h5),
.markdown-content :deep(h6) {
  margin-top: 1.8em;
  margin-bottom: 0.8em;
  font-weight: 600;
  color: #343a40; /* Darker headings */
  padding-bottom: 0;
  border-bottom: none;
}
.markdown-content :deep(h1) { font-size: 1.8rem; }
.markdown-content :deep(h2) { font-size: 1.5rem; border-bottom: 1px solid #eee; padding-bottom: 0.3em; } /* Add line under h2 */
.markdown-content :deep(h3) { font-size: 1.3rem; }
.markdown-content :deep(p) {
  margin-bottom: 1.2em; /* Slightly more spacing */
  color: #495057; /* Slightly lighter text for paragraphs */
}
.markdown-content :deep(code) { /* Inline code */
  background-color: #e9ecef;
  padding: 0.2em 0.45em;
  margin: 0 0.1em;
  font-size: 88%; /* Slightly larger inline code */
  border-radius: 4px;
  font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  color: #c7254e; /* Use a distinct color for inline code */
  word-break: break-word;
  text-align: left;
}
.markdown-content :deep(pre) { /* Code blocks */
  background-color: #f8f9fa; /* Lighter background for code blocks */
  border: 1px solid #dee2e6; /* Add border */
  color: #343a40; /* Darker text for code */
  padding: 1.2em; /* More padding */
  margin-bottom: 1.6em;
  border-radius: 6px;
  overflow-x: auto; /* Keep horizontal scroll */
  font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  text-align: left;
}
.markdown-content :deep(pre code) { /* Code inside pre */
  background-color: transparent;
  padding: 0;
  margin: 0;
  font-size: inherit;
  color: inherit;
  border-radius: 0;
  border: none; /* No border for code inside pre */
  text-align: left;
}
.markdown-content :deep(blockquote) {
   border-left: 5px solid #007bff; /* Primary color border */
   padding: 0.8rem 1.2rem; /* Adjust padding */
   margin-left: 0;
   margin-right: 0;
   margin-bottom: 1.2em;
   background-color: #f8f9fa; /* Slight background */
   color: #495057; /* Text color */
}
.markdown-content :deep(blockquote p) { /* Paragraphs inside blockquote */
    margin-bottom: 0.5em; /* Reduce bottom margin */
    color: #495057;
}
.markdown-content :deep(a) {
   color: var(--primary-color);
   text-decoration: none;
   font-weight: 500; /* Slightly bolder links */
}
.markdown-content :deep(a:hover) {
   text-decoration: underline;
}
.markdown-content :deep(hr) {
   border: 0;
   height: 2px; /* Thicker hr */
   background-color: #e9ecef; /* Lighter hr */
   margin: 2.5em 0;
}
.markdown-content :deep(ul), .markdown-content :deep(ol) {
    padding-left: 2em; /* Indent lists */
    margin-bottom: 1.2em;
}
.markdown-content :deep(li) {
    margin-bottom: 0.5em; /* Space between list items */
}
.markdown-content :deep(table) {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1.5em;
    border: 1px solid #dee2e6;
}
.markdown-content :deep(th), .markdown-content :deep(td) {
    border: 1px solid #dee2e6;
    padding: 0.6em 0.8em;
    text-align: left;
}
.markdown-content :deep(th) {
    background-color: #f8f9fa;
    font-weight: 600;
}
.markdown-content :deep(tr:nth-child(even)) {
    background-color: #f8f9fa; /* Subtle striping */
}


/* --- Responsive Adjustments --- */
@media (max-width: 768px) {
  .header-section { flex-direction: column; align-items: flex-start; gap: 1rem; }
  .header-right { width: 100%; justify-content: flex-start; flex-wrap: wrap; }
  .description-section { flex-direction: column; }
  .image-container { max-width: 100%; max-height: 250px; }
  .image-container :deep(img) { max-height: 200px; }
  .charts-grid { grid-template-columns: 1fr; }
  .correlation-table { font-size: 0.8rem; }
  .correlation-table :deep(td), .correlation-table :deep(th) { padding: 0.4rem 0.2rem !important; }
  .view-toggle-container { justify-content: center; }
  .single-chart-card :deep(.p-card-content) { min-height: 350px; }
}

</style>