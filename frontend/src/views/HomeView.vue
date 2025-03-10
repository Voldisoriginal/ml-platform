<template>
    <div class="home">
        <h1>ML Platform</h1>
        <DatasetUpload @dataset-uploaded="handleDatasetUploaded" />
        <TrainSettings v-if="datasetFilename" :columns="datasetColumns" @settings-submitted="updateTrainSettings" />
        <ModelSelection v-if="datasetFilename" :models="availableModels" @model-selected="handleModelSelected" />
        <button v-if="isFormComplete" @click="trainModel" :disabled="isTraining">Train Model</button>
        <TrainingProgress v-if="isTraining || trainingResult" :is-training="isTraining" :result="trainingResult"
            :taskId="taskId" />
        <!-- Удалена TrainedModelsList отсюда -->

        <ModelInference v-if="selectedModelForInference" :modelId="selectedModelForInference"
            @close-inference="selectedModelForInference = null" :featureNames="featureNames" />
    </div>
</template>

<script>
    import DatasetUpload from '@/components/DatasetUpload.vue';
    import TrainSettings from '@/components/TrainSettings.vue';
    import ModelSelection from '@/components/ModelSelection.vue';
    import TrainingProgress from '@/components/TrainingProgress.vue';
    // import TrainedModelsList from '@/components/TrainedModelsList.vue'; // Удалено
    import ModelInference from '@/components/ModelInference.vue';
    import axios from 'axios';


    const API_BASE_URL = 'http://localhost:8000'; // URL вашего FastAPI сервера

    export default {
        name: 'HomeView',
        components: {
            DatasetUpload,
            TrainSettings,
            ModelSelection,
            TrainingProgress,
            // TrainedModelsList, // Удалено
            ModelInference
        },
        data() {
            return {
                datasetFilename: null,
                datasetColumns: [],
                targetColumn: null,
                trainSettings: {
                    train_size: 0.7,
                    random_state: 42,
                },
                selectedModel: null,  // { model_type: '...', params: {} }
                availableModels: [  // Доступные модели (пока хардкодим)
                    { type: 'LinearRegression', name: 'Linear Regression', params: {} },
                    {
                        type: 'DecisionTreeRegressor', name: 'Decision Tree', params: { max_depth: 5, min_samples_split: 2 }
                    },
                ],
                isTraining: false,
                trainingResult: null, // Результат обучения (метрики и т.д.)
                // trainedModels: [], // Удалено
                selectedModelForInference: null, // ID модели для инференса
                featureNames: [], // Названия признаков для выбранной модели
                taskId: null,  // ID задачи Celery
            };
        },
        computed: {
            isFormComplete() {
                return this.datasetFilename && this.targetColumn && this.selectedModel;
            }
        },
        methods: {
            async handleDatasetUploaded(data) {
                this.datasetFilename = data.filename;
                this.datasetColumns = data.columns;
                // сбрасываем прошлые значения
                this.targetColumn = null;
                this.selectedModel = null;
                this.trainingResult = null;

            },
            updateTrainSettings(settings) {
                this.targetColumn = settings.targetColumn;
                this.trainSettings = {
                    train_size: settings.trainSize,
                    random_state: settings.randomState
                }
            },
            handleModelSelected(model) {
                this.selectedModel = model;
            },
            async trainModel() {
                if (!this.isFormComplete) return;

                this.isTraining = true;
                this.trainingResult = null;
                this.taskId = null; // Сбрасываем task id
                try {
                    const response = await axios.post(`${API_BASE_URL}/train/`, new URLSearchParams({
                        dataset_filename: this.datasetFilename,
                        target_column: this.targetColumn,
                        train_settings: JSON.stringify(this.trainSettings),
                        model_params: JSON.stringify(this.selectedModel),
                    }));

                    // this.trainingResult = response.data; //  НЕ сразу результат, а ID задачи
                    this.taskId = response.data.task_id; //  Получаем ID задачи

                    // await this.fetchTrainedModels(); //  НЕ обновляем список моделей сразу!
                } catch (error) {
                    console.error("Error during training:", error);
                    if (error.response) {
                        alert(`Error: ${error.response.data.detail}`);
                    } else {
                        alert('An error occurred during training.');
                    }
                } finally {
                    // this.isTraining = false; // НЕ ставим false сразу!
                }
            },
            // async fetchTrainedModels() { //Удалено
            //     try {
            //         const response = await axios.get(`${API_BASE_URL}/trained_models/`);
            //         this.trainedModels = response.data;
            //     } catch (error) {
            //         console.error("Error fetching trained models:", error);
            //     }
            // },
            async startInference(modelId) {
                this.selectedModelForInference = modelId;
                // Получаем названия признаков:
                try {
                    const response = await axios.get(`${API_BASE_URL}/features/${modelId}`);
                    this.featureNames = response.data;
                } catch (error) {
                    console.error("Error fetching features:", error);
                    this.featureNames = []; //  На случай ошибки
                }
            },
        },
        // async created() { //Удалено
        //     await this.fetchTrainedModels();
        // }

    };
</script>

<style scoped>
    /* Стили, специфичные для этого компонента */
    .home {
        padding: 20px;
    }
</style>
