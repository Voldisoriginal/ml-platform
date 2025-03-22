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
              <FileUpload ref="newDatasetUpload" mode="basic" :customUpload="true" @uploader="handleNewDatasetUpload" accept=".csv" :auto="false" chooseLabel="Select CSV" />
         </div>
         <div  v-if="previewContent" style="overflow-x: auto; margin-bottom: 1rem">
               <pre>{{ previewContent }}</pre>
          </div>
          <div class="p-field">
             <label for="new-dataset-name">Dataset Name</label>
             <InputText id="new-dataset-name" v-model="newDatasetName" required />
           </div>
  
        <Button label="Continue" :disabled="!newDatasetName || !newDatasetFile" @click="continueWithNewDataset" />
      </div>
  
      <!-- Train Settings and Model Selection (Conditional) -->
         <div v-if="datasetFilename">
           <TrainSettings :columns="datasetColumns" @settings-submitted="updateTrainSettings" />
           <ModelSelection :models="availableModels" @model-selected="handleModelSelected" />
           <Button v-if="isFormComplete" @click="trainModel" :disabled="isTraining">Train Model</Button>
         </div>
  
         <TrainingProgress
    v-model:is-training="isTraining"
    v-model:result="trainingResult"
    :task-id="taskId"
    @training-complete="handleTrainingComplete"
  />
         <!-- Inference Component (Remains unchanged) -->
       <ModelInference v-if="selectedModelForInference" :modelId="selectedModelForInference" @close-inference="selectedModelForInference = null" :featureNames="featureNames"/>
    </div>
  </template>
  
  <script setup>
  import { ref, computed, reactive, onMounted } from 'vue';
  import { useRouter } from 'vue-router';
  import axios from 'axios';
  import TrainSettings from '@/components/TrainSettings.vue';
  import ModelSelection from '@/components/ModelSelection.vue';
  import TrainingProgress from '@/components/TrainingProgress.vue';
  import ModelInference from '@/components/ModelInference.vue';
  import Button from 'primevue/button';
  import FileUpload from 'primevue/fileupload';
  import InputText from 'primevue/inputtext';  // Import InputText
  import { parse } from 'papaparse';
  import { useToast } from "primevue/usetoast"; //Use toast
  const API_BASE_URL = 'http://localhost:8000'; // URL вашего FastAPI сервера
  
  const toast = useToast();
  const router = useRouter();
  const uploadMode = ref(null); // 'new' or null
  const newDatasetFile = ref(null);
  const newDatasetName = ref('');  // For the new dataset name
  const datasetFilename = ref(null);
  const datasetColumns = ref([]);
  const targetColumn = ref(null);
  const trainSettings = reactive({
    train_size: 0.7,
    random_state: 42,
  });
  const selectedModel = ref(null);
  const availableModels = ref([
      { type: 'LinearRegression', name: 'Linear Regression', params: {} },
                      {
                          type: 'DecisionTreeRegressor', name: 'Decision Tree', params: { max_depth: 5, min_samples_split: 2 }
                      },
  ]);
  const isTraining = ref(false);
  const trainingResult = ref(null);
  const taskId = ref(null);
  const selectedModelForInference = ref(null);
  const featureNames = ref([]);
  
  const previewContent = ref(null);
  
  
  const isFormComplete = computed(() => {
    return datasetFilename.value && targetColumn.value && selectedModel.value;
  });
  
  const chooseExisting = () => {
    router.push('/datasets');
  };
  
  const handleNewDatasetUpload = async (event) => {   //  Handles the new dataset upload
      newDatasetFile.value = event.files[0];
       if (newDatasetFile.value) {
          // Load preview
          const text = await newDatasetFile.value.text();
          parse(text, {
              preview: 10,
              complete: (results) => {
                  previewContent.value = results.data.map(row => row.join(',')).join('\n');
              },
              error: (error) => {
                  console.error("CSV parsing error:", error);
                  previewContent.value = "Error loading preview.";
              }
        });
      }
  };
  const continueWithNewDataset = async () => {
      if (!newDatasetFile.value || !newDatasetName.value) {
      return; // Prevent if file or name is missing
    }
      const formData = new FormData();
      formData.append('file', newDatasetFile.value);
      formData.append('name', newDatasetName.value); //  Append the dataset name
  
       try {
         const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
           headers: {
                 'Content-Type': 'multipart/form-data'
               }
         });
         // Success
         toast.add({severity: 'success', summary: 'Success', detail: 'Dataset uploaded successfully!', life: 3000});
          datasetFilename.value = response.data.filename;   //  Set the dataset filename
          datasetColumns.value = response.data.columns; // and columns
          resetNewDatasetForm();  // Reset
  
       } catch (error) {
           console.error("Error uploading new dataset:", error);         
           const detail = error.response?.data?.detail || "Failed to upload dataset";
           toast.add({severity: 'error', summary: 'Error', detail: detail, life: 5000});
       }
  };
  
  
  const resetNewDatasetForm = () => {
     newDatasetFile.value = null;
      newDatasetName.value = '';  // Reset name
      previewContent.value = null;
      if (this.$refs.newDatasetUpload) {
        this.$refs.newDatasetUpload.clear();
      }
  
  };
  
  
  const updateTrainSettings = (settings) => {
    targetColumn.value = settings.targetColumn;
    Object.assign(trainSettings, {
      train_size: settings.trainSize,
      random_state: settings.randomState,
    });
  };
  
  const handleModelSelected = (model) => {
    selectedModel.value = model;
  };

  
const handleTrainingComplete = (result) => {
    // No need to set isTraining or trainingResult here,
    // because they are already updated via v-model.
    // You can do any *additional* actions here,
    // like showing a success message.

    console.log("Training complete!", result);
}
  
  const trainModel = async () => {
    if (!isFormComplete.value) return;
  
    isTraining.value = true;
    trainingResult.value = null;
    taskId.value = null;
    try {
      const response = await axios.post(`${API_BASE_URL}/train/`, new URLSearchParams({
        dataset_filename: datasetFilename.value,
        target_column: targetColumn.value,
        train_settings: JSON.stringify(trainSettings),
        model_params: JSON.stringify(selectedModel.value),
      }));
      taskId.value = response.data.task_id;
    } catch (error) {
      console.error("Error during training:", error);
      if (error.response) {
          toast.add({severity: 'error', summary: 'Error', detail: error.response.data.detail, life: 3000});
      } else {
        toast.add({severity: 'error', summary: 'Error', detail: "An error occurred during training", life: 3000});
      }
    }
  };
  
  const startInference = async (modelId) => {
    selectedModelForInference.value = modelId;
    try {
      const response = await axios.get(`${API_BASE_URL}/features/${modelId}`);
      featureNames.value = response.data;
    } catch (error) {
      console.error("Error fetching features:", error);
      featureNames.value = [];
    }
  };
  //For load selected dataset from DatasetsPage
  onMounted(() => {
       if (router.currentRoute.value.query.dataset) {       
          loadSelectedDataset(router.currentRoute.value.query.dataset);
       }
  });
  const loadSelectedDataset = async(datasetId) => {
      try {
       const response = await axios.get(`${API_BASE_URL}/dataset/${datasetId}`);
         if (response.data && response.data.filename && response.data.columns) {
              datasetFilename.value = response.data.filename;
              datasetColumns.value = response.data.columns;
              // Reset the targetColumn, selectedModel and training result.
              targetColumn.value = null;
              selectedModel.value = null;
              trainingResult.value= null;
          }
  
      }catch(error){
          console.error("Failed to load selected dataset", error);
      }
  }
  </script>
  
  <style scoped>
  /* General Home Styles */
  .home {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }
  
  /* Dataset selection buttons */
  .dataset-selection {
    display: flex;
    gap: 1rem; /* Space between the buttons */
    margin-bottom: 2rem; /* Space below the buttons */
  }
  
  .p-button-raised {
    /* Your existing button styles */
  }
  
  /* Upload New Dataset Section */
  .upload-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%; /* Full width */
    max-width: 600px; /* Limit the maximum width */
    margin-bottom: 2rem; /* Space below this section */
  }
  
  .p-field {
    width: 100%; /* Fields take full width of container */
    margin-bottom: 1rem;
  }
  /* Style for other components TrainSettings, ModelSelection*/
  </style>
  