<template>
  <div>
    <h1>Trained Models</h1>
    <TrainedModelsList :models="trainedModels" @model-selected-for-inference="startInference" @show-model-details="showModelDetails" />
    <ModelInference v-if="selectedModelForInference" :modelId="selectedModelForInference"
                    @close-inference="selectedModelForInference = null" :featureNames="featureNames"/>
    <ModelDetailsModal v-if="selectedModel" :model="selectedModel" @close="selectedModel = null" />
  </div>
</template>

<script>
import axios from 'axios';
import TrainedModelsList from '@/components/TrainedModelsList.vue';
import ModelInference from '@/components/ModelInference.vue';
import ModelDetailsModal from '@/components/ModelDetailsModal.vue'; // Import the modal

const API_BASE_URL = 'http://localhost:8000';

export default {
  components: {
    TrainedModelsList,
    ModelInference,
    ModelDetailsModal, // Register the modal
  },
  data() {
    return {
      trainedModels: [],
      selectedModelForInference: null,
      featureNames: [],
      selectedModel: null, // Add this
    };
  },
  methods: {
    async fetchTrainedModels() {
      try {
        let url = `${API_BASE_URL}/trained_models/search_sort`;
        const queryParams = [];

        if (this.searchQuery) {
          queryParams.push(`search_query=${this.searchQuery}`);
        }
        if (this.selectedSortBy) {
          queryParams.push(`sort_by=${this.selectedSortBy}`);
        }
        if (this.selectedSortOrder) {
          queryParams.push(`sort_order=${this.selectedSortOrder}`);
        }
        if (this.selectedModelType) {
          queryParams.push(`model_type=${this.selectedModelType}`);
        }
        if (this.selectedDatasetFilename) {
          queryParams.push(`dataset_filename=${this.selectedDatasetFilename}`);
        }

        if (queryParams.length > 0) {
          url += `?${queryParams.join('&')}`;
        }
        const response = await axios.get(url);
        this.trainedModels = response.data;
      } catch (error) {
        console.error('Error fetching trained models:', error);
      }
    },
    async startInference(modelId) {
      this.selectedModelForInference = modelId;
      try {
        const response = await axios.get(`${API_BASE_URL}/features/${modelId}`);
        this.featureNames = response.data;
      } catch (error) {
        console.error("Error fetching features:", error);
        this.featureNames = [];
      }
    },
    showModelDetails(model) {
      this.selectedModel = model;
    },
  },
  created() {
    this.fetchTrainedModels();
  },
  computed: {},
  watch: {
    searchQuery() {
      this.fetchTrainedModels();
    },
    selectedSortBy() {
      this.fetchTrainedModels();
    },
    selectedSortOrder() {
      this.fetchTrainedModels();
    },
    selectedModelType() {
      this.fetchTrainedModels();
    },
    selectedDatasetFilename() {
      this.fetchTrainedModels();
    },
  },
};
</script>
