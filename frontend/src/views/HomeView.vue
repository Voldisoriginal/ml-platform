<template>
  <div class="home">
    <h1>ML Platform</h1>

    <div class="dataset-selection">
      <Button label="Upload New Dataset" @click="uploadMode = 'new'" class="p-button-raised" />
      <Button label="Choose from Existing Datasets" @click="chooseExisting" class="p-button-raised p-button-secondary" />
    </div>

    <!-- Upload New Dataset Section -->
    <div v-if="uploadMode === 'new'" class="upload-section">
      <div class="p-field">
           <label for="new-dataset-file">Dataset File (CSV)</label>
           <!-- Используем :key чтобы принудительно пересоздать компонент при сбросе -->
            <FileUpload
               ref="newDatasetUpload"
               name="file"
               :key="fileUploadKey"
               mode="basic"
               :customUpload="true"
               @select="handleNewDatasetSelected"
               @clear="handleNewDatasetCleared"
               accept=".csv"
               :auto="false"
               chooseLabel="Select CSV"
               :maxFileSize="50000000" /> <!-- Ограничение 50MB -->
       </div>
       <div v-if="previewContent" class="preview-section p-card">
           <div class="p-card-title">Preview (first 10 rows)</div>
           <div class="p-card-content">
               <ScrollPanel style="width: 100%; height: 200px">
                  <pre>{{ previewContent }}</pre>
               </ScrollPanel>
           </div>
       </div>
        <div class="p-field">
           <label for="new-dataset-name">Dataset Name</label>
           <InputText id="new-dataset-name" v-model="newDatasetName" required />
         </div>

      <Button label="Upload & Continue" :disabled="!newDatasetName || !newDatasetFile" @click="continueWithNewDataset" icon="pi pi-upload" class="p-mt-2"/>
      <Button label="Cancel" @click="cancelNewDataset" icon="pi pi-times" class="p-button-secondary p-mt-2 p-ml-2"/>

    </div>

    <!-- Settings and Selection (Отображается ПОСЛЕ выбора датасета) -->
    <div v-if="datasetFilename" class="pipeline-config p-mt-4">
        <div class="p-card">
             <div class="p-card-title">Configuration for: {{ datasetFilename }}</div>
             <div class="p-card-content">
                 <TrainSettings
                     :columns="datasetColumns"
                     @settings-submitted="updateTrainSettings"
                     :key="datasetFilename" 
                  />
                 <!-- Новый ModelSelection -->
                 <ModelSelection
                      v-if="availableModels.length > 0"
                     :models="availableModels"
                     @model-selected="handleModelSelected"
                     :key="datasetFilename + '-models'" 
                 />
                 <div v-else class="p-text-center p-p-3">
                      Loading available models...
                       <ProgressSpinner style="width:30px; height:30px" strokeWidth="8" />
                 </div>
             </div>
        </div>
         <div class="train-button-container p-mt-3 p-d-flex p-jc-center">
             <Button
                 label="Train Model"
                 icon="pi pi-play"
                 @click="trainModel"
                 :disabled="!isFormComplete || isTraining"
                 :loading="isTraining"
                 class="p-button-lg"
              />
          </div>
    </div>
    <!-- Остальное без изменений -->
     <TrainingProgress
          v-if="isTraining || taskId"
          v-model:is-training="isTraining"
          v-model:result="trainingResult"
          :task-id="taskId"
          @training-complete="handleTrainingComplete"
      />

     <ModelInference
          v-if="selectedModelForInference"
          :modelId="selectedModelForInference"
          @close-inference="selectedModelForInference = null"
          :featureNames="featureNames"
      />
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router'; // Добавим useRoute
import axios from 'axios';
import TrainSettings from '@/components/TrainSettings.vue';
import ModelSelection from '@/components/ModelSelection.vue'; // Импортируем обновленный
import TrainingProgress from '@/components/TrainingProgress.vue';
import ModelInference from '@/components/ModelInference.vue';
import Button from 'primevue/button';
import FileUpload from 'primevue/fileupload';
import InputText from 'primevue/inputtext';
import ScrollPanel from 'primevue/scrollpanel'; // Для превью
import ProgressSpinner from 'primevue/progressspinner'; // Для индикатора загрузки моделей
import { parse } from 'papaparse';
import { useToast } from "primevue/usetoast";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'; // Используем переменную окружения

const toast = useToast();
const router = useRouter();
const route = useRoute(); // Получаем текущий роут

// --- State for Dataset Handling ---
const uploadMode = ref(null); // 'new' or null
const newDatasetFile = ref(null);
const newDatasetName = ref('');
const previewContent = ref(null);
const fileUploadKey = ref(0); // Для сброса FileUpload

// --- State for Pipeline Configuration ---
const datasetFilename = ref(null); // Имя файла выбранного/загруженного датасета
const datasetColumns = ref([]); // Колонки выбранного/загруженного датасета
const targetColumn = ref(null);
const trainSettings = reactive({
  train_size: 0.7,
  random_state: 42,
});
const selectedModel = ref(null); // { model_type: '...', params: {...} }
const availableModels = ref([]); // Массив моделей с бэкенда
const isLoadingModels = ref(false); // Флаг загрузки моделей

// --- State for Training ---
const isTraining = ref(false);
const trainingResult = ref(null); // Результат успешного обучения
const taskId = ref(null); // ID запущенной задачи Celery

// --- State for Inference ---
const selectedModelForInference = ref(null);
const featureNames = ref([]); // Фичи для инференса

// --- Computed Properties ---
const isFormComplete = computed(() => {
  return datasetFilename.value &&
         targetColumn.value &&
         selectedModel.value &&
         selectedModel.value.model_type; // Убедимся, что модель выбрана
});

// --- Methods ---

// Dataset Selection/Upload
const chooseExisting = () => {
  router.push('/datasets');
};

const handleNewDatasetSelected = async (event) => {
  if (!event.files || event.files.length === 0) return;
  newDatasetFile.value = event.files[0];

  // Generate Preview
  try {
    const text = await newDatasetFile.value.text();
    parse(text, {
      preview: 10, // Количество строк для превью
      header: true, // Используем заголовки
      skipEmptyLines: true,
      complete: (results) => {
        if (results.data && results.data.length > 0) {
            // Форматируем для pre
            const header = results.meta.fields.join(',');
            const rows = results.data.map(row => results.meta.fields.map(field => row[field] ?? '').join(','));
            previewContent.value = header + '\n' + rows.join('\n');
        } else {
             previewContent.value = "Could not parse preview (is the file empty or invalid?).";
        }
      },
      error: (error) => {
        console.error("CSV parsing error:", error);
        previewContent.value = `Error loading preview: ${error.message}`;
        toast.add({ severity: 'warn', summary: 'Preview Error', detail: `Could not parse CSV for preview: ${error.message}`, life: 4000 });
      }
    });
  } catch (e) {
     console.error("Error reading file for preview:", e);
     previewContent.value = "Error reading file.";
     toast.add({ severity: 'error', summary: 'File Read Error', detail: 'Could not read the selected file.', life: 3000 });
     resetNewDatasetForm(); // Сбросить, если файл не читается
  }
};

const handleNewDatasetCleared = () => {
   newDatasetFile.value = null;
   previewContent.value = null;
   // Не сбрасываем имя здесь
}

const continueWithNewDataset = async () => {
  if (!newDatasetFile.value || !newDatasetName.value) {
    toast.add({ severity: 'warn', summary: 'Missing Info', detail: 'Please select a CSV file and provide a dataset name.', life: 3000 });
    return;
  }

  const formData = new FormData();
  formData.append('file', newDatasetFile.value, newDatasetFile.value.name); // Явно передаем имя файла
  formData.append('name', newDatasetName.value);
  // Добавить другие поля (description, author), если они есть во фронтенде

  try {
    const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    toast.add({ severity: 'success', summary: 'Success', detail: 'Dataset uploaded successfully!', life: 3000 });
    // --- Устанавливаем состояние для следующего шага ---
    datasetFilename.value = response.data.filename;
    datasetColumns.value = response.data.columns;
    // --- Сбрасываем форму загрузки ---
    resetNewDatasetForm();
    uploadMode.value = null; // Закрываем секцию загрузки
    // --- Загружаем доступные модели ---
    fetchAvailableModels(); // Загружаем модели после успешной загрузки датасета

  } catch (error) {
    console.error("Error uploading new dataset:", error);
    const detail = error.response?.data?.detail || "Failed to upload dataset. Check server logs.";
    toast.add({ severity: 'error', summary: 'Upload Error', detail: detail, life: 5000 });
  }
};

const cancelNewDataset = () => {
   resetNewDatasetForm();
   uploadMode.value = null;
}

const resetNewDatasetForm = () => {
  newDatasetFile.value = null;
  newDatasetName.value = '';
  previewContent.value = null;
  fileUploadKey.value++; // Изменяем ключ для пересоздания FileUpload
};

// --- Fetching Available Models ---
const fetchAvailableModels = async () => {
   isLoadingModels.value = true;
   availableModels.value = []; // Очищаем перед загрузкой
   try {
       const response = await axios.get(`${API_BASE_URL}/available_models/`);
       availableModels.value = response.data;
       console.log("Fetched available models:", availableModels.value);
   } catch (error) {
       console.error("Error fetching available models:", error);
       toast.add({ severity: 'error', summary: 'Model Load Error', detail: 'Could not load available models from server.', life: 5000 });
       availableModels.value = []; // Оставляем пустым в случае ошибки
   } finally {
       isLoadingModels.value = false;
   }
};


// Pipeline Configuration
const updateTrainSettings = (settings) => {
  targetColumn.value = settings.targetColumn;
  trainSettings.train_size = settings.trainSize;
  trainSettings.random_state = settings.randomState;
   console.log("Train settings updated:", { targetColumn: targetColumn.value, ...trainSettings });
};

const handleModelSelected = (model) => {
  // model здесь { model_type: '...', params: {...} } или null
  selectedModel.value = model;
   console.log("Model selected in HomeView:", JSON.parse(JSON.stringify(model)));
};

// --- Training ---
const trainModel = async () => {
  if (!isFormComplete.value || isTraining.value) return;

  isTraining.value = true;
  trainingResult.value = null; // Сбрасываем предыдущий результат
  taskId.value = null; // Сбрасываем предыдущий task ID

  // Подготовка данных для отправки
  // Axios ожидает данные формы или объект для application/json
  // FastAPI ожидает Form data. Используем URLSearchParams
  const formData = new URLSearchParams();
  formData.append('dataset_filename', datasetFilename.value);
  formData.append('target_column', targetColumn.value);
  formData.append('train_settings', JSON.stringify(trainSettings));
  formData.append('model_params', JSON.stringify(selectedModel.value)); // selectedModel уже содержит { model_type, params }

   console.log("Submitting training request with:", {
       dataset_filename: datasetFilename.value,
       target_column: targetColumn.value,
       train_settings: JSON.stringify(trainSettings),
       model_params: JSON.stringify(selectedModel.value)
   });


  try {
    const response = await axios.post(`${API_BASE_URL}/train/`, formData); // Отправляем как form data
    taskId.value = response.data.task_id;
    console.log("Training task started with ID:", taskId.value);
    toast.add({ severity: 'info', summary: 'Training Started', detail: `Task ID: ${taskId.value}`, life: 3000 });
    // TrainingProgress компонент начнет опрос статуса по taskId
  } catch (error) {
    console.error("Error starting training:", error);
    const detail = error.response?.data?.detail || "Failed to start training task. Check server logs.";
    toast.add({ severity: 'error', summary: 'Training Error', detail: detail, life: 6000 });
    isTraining.value = false; // Сбрасываем флаг, если запуск не удался
    taskId.value = null;
  }
};

const handleTrainingComplete = (result) => {
   // Этот метод вызывается компонентом TrainingProgress, когда статус задачи SUCCESS
   // isTraining и trainingResult обновляются через v-model
   // Дополнительные действия здесь:
   console.log("Training complete! Result received in HomeView:", result);
   toast.add({ severity: 'success', summary: 'Training Successful', detail: `Model ${result?.id || ''} trained. Metrics: ${JSON.stringify(result?.metrics || {})}`, life: 5000 });
   // Можно, например, предложить запустить инференс для только что обученной модели
   // startInference(result.id); // Если нужно сразу перейти к инференсу
};


// Inference (старый код без изменений)
const startInference = async (modelId) => {
  selectedModelForInference.value = modelId;
  try {
    const response = await axios.get(`${API_BASE_URL}/features/${modelId}`);
    featureNames.value = response.data;
  } catch (error) {
    console.error("Error fetching features:", error);
    featureNames.value = [];
     toast.add({ severity: 'error', summary: 'Inference Error', detail: 'Could not fetch features for inference.', life: 4000 });
  }
};


// --- Lifecycle and Route Handling ---

// Загрузка данных при монтировании и при изменении query параметра
onMounted(() => {
  const datasetIdFromQuery = route.query.dataset;
  if (datasetIdFromQuery) {
    console.log("Dataset ID found in query:", datasetIdFromQuery);
    loadSelectedDataset(datasetIdFromQuery);
  } else {
     // Если нет датасета в query, можно загрузить модели сразу
     // или подождать выбора/загрузки датасета
     fetchAvailableModels();
  }
});

// Следим за изменениями query параметра 'dataset'
watch(() => route.query.dataset, (newDatasetId) => {
   if (newDatasetId && newDatasetId !== datasetFilename.value) { // Проверяем, что ID действительно новый
       console.log("Dataset ID changed in query:", newDatasetId);
       loadSelectedDataset(newDatasetId);
   } else if (!newDatasetId) {
        // Если параметр 'dataset' удален из URL, сбрасываем состояние
        console.log("Dataset query parameter removed. Resetting state.");
        resetPipelineConfig();
   }
});

const loadSelectedDataset = async (datasetIdentifier) => {
    // datasetIdentifier может быть ID или filename
    console.log(`Attempting to load dataset: ${datasetIdentifier}`);
    // Сбрасываем предыдущее состояние перед загрузкой нового
    resetPipelineConfig();
    isLoadingModels.value = true; // Показываем загрузку

    try {
        // Сначала попробуем получить по ID, если он похож на UUID
        let datasetData = null;
        const isLikelyUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(datasetIdentifier);

        if (isLikelyUuid) {
            try {
               const response = await axios.get(`${API_BASE_URL}/dataset/${datasetIdentifier}`);
               datasetData = response.data;
            } catch (error) {
                 if (error.response?.status === 404) {
                     logger.warning(`Dataset not found by ID ${datasetIdentifier}, will try by filename.`);
                 } else {
                      throw error; // Пробрасываем другие ошибки
                 }
            }
        }

        // Если по ID не нашли или это был не UUID, пробуем по имени файла (на всякий случай)
        // Но лучше всегда работать с ID
        if (!datasetData) {
           // В реальном приложении лучше передавать ID с страницы датасетов
           console.warn("Loading dataset by filename is less reliable. Please use dataset ID.");
            // TODO: Нужен эндпоинт для поиска датасета по имени файла, если это необходимо
            // Пока что предполагаем, что передан ID
            if (!isLikelyUuid) {
                 throw new Error("Invalid dataset identifier provided. Expected UUID.");
            } else {
                 // Если был UUID, но 404, значит датасет не найден
                 throw new Error(`Dataset with ID ${datasetIdentifier} not found.`);
            }
        }

       // Успешно получили данные датасета
       if (datasetData && datasetData.filename && datasetData.columns) {
            datasetFilename.value = datasetData.filename; // Используем filename из ответа
            datasetColumns.value = datasetData.columns;
            toast.add({ severity: 'success', summary: 'Dataset Loaded', detail: `Loaded ${datasetData.name || datasetFilename.value}`, life: 3000 });
            // Загружаем модели ПОСЛЕ успешной загрузки данных датасета
            await fetchAvailableModels();
        } else {
            throw new Error("Received invalid dataset data from server.");
        }

    } catch(error) {
        console.error("Failed to load selected dataset:", error);
        const detail = error.response?.data?.detail || error.message || "Could not load dataset details.";
        toast.add({ severity: 'error', summary: 'Dataset Load Failed', detail: detail, life: 5000 });
        resetPipelineConfig(); // Сбрасываем в случае ошибки
        // Очищаем query параметр, если загрузка не удалась? (Опционально)
        // router.replace({ query: { ...route.query, dataset: undefined } });
    } finally {
        isLoadingModels.value = false;
    }
}

// Функция для сброса конфигурации пайплайна
const resetPipelineConfig = () => {
     datasetFilename.value = null;
     datasetColumns.value = [];
     targetColumn.value = null;
     selectedModel.value = null;
     trainingResult.value = null;
     taskId.value = null;
     isTraining.value = false;
     availableModels.value = []; // Очищаем список моделей
     // Сбросить состояние TrainSettings?
     // trainSettings.train_size = 0.7;
     // trainSettings.random_state = 42;
}

</script>

<style scoped>
/* General Home Styles */
.home {
  max-width: 1200px; /* Ограничим максимальную ширину */
  margin: 0 auto; /* Центрируем */
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center; /* Центрируем контент по горизонтали */
}

.home h1 {
  color: var(--primary-color);
}

/* Dataset selection buttons */
.dataset-selection {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  flex-wrap: wrap; /* Перенос кнопок на новую строку */
  justify-content: center;
}

/* Upload New Dataset Section */
.upload-section {
  display: flex;
  flex-direction: column;
  align-items: stretch; /* Растягиваем элементы по ширине */
  width: 100%;
  max-width: 600px;
  margin-bottom: 2rem;
  padding: 2rem;
  border: 1px solid var(--surface-d);
  border-radius: var(--border-radius);
  background-color: var(--surface-a);
}

.p-field {
  width: 100%;
  margin-bottom: 1.5rem; /* Увеличим отступ */
}

.p-field > label {
   display: block;
   margin-bottom: 0.5rem;
   font-weight: bold;
}

.preview-section {
   margin-bottom: 1.5rem;
   border: 1px solid var(--surface-d);
}
.preview-section pre {
   background-color: var(--surface-b);
   color: var(--text-color);
   padding: 1rem;
   border-radius: var(--border-radius);
   font-family: monospace;
   white-space: pre;
   overflow-x: auto; /* Добавим горизонтальный скролл для превью */
}

.preview-section .p-card-title {
   font-size: 1rem;
   margin-bottom: 0.5rem;
}

/* Pipeline Configuration Section */
.pipeline-config {
   width: 100%;
   max-width: 900px; /* Можно сделать шире, чем загрузка */
}

.pipeline-config .p-card-title {
    font-size: 1.2rem;
    color: var(--primary-color);
    border-bottom: 1px solid var(--surface-d);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Train Button */
.train-button-container {
   width: 100%;
}


/* Общие стили для кнопок */
.p-button {
   margin-right: 0.5rem;
}
.p-button-lg {
   padding: 0.75rem 1.5rem;
   font-size: 1.1rem;
}

/* Индикатор загрузки моделей */
.p-text-center.p-p-3 {
   display: flex;
   flex-direction: column;
   align-items: center;
   justify-content: center;
   min-height: 100px; /* Чтобы было видно */
   color: var(--text-color-secondary);
}
</style>