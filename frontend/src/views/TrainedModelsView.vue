<template>
  <div>
    <h1>Trained Models</h1>
    <TrainedModelsList :models="trainedModels"
                       :items-per-page="itemsPerPageOptions"
                       @model-selected-for-inference="startInference"
                       @show-model-details="showModelDetails"
                       @fetch-models="fetchTrainedModels"
                       ref="modelsList"

    />
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
      itemsPerPageOptions: [5, 10, 20, 50], // Add more options if needed
    };
  },
  methods: {
    async fetchTrainedModels() {
      try {
          let url = `${API_BASE_URL}/trained_models/search_sort`;

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
      setItemsPerPage(value) {
          if (this.$refs.modelsList)
          {
              this.$refs.modelsList.itemsPerPage = parseInt(value, 10)
              this.$refs.modelsList.currentPage = 1;// Reset to the first page when the items per page changes

          }
      },
  },
  created() {
    this.fetchTrainedModels();
  },
};
</script>
