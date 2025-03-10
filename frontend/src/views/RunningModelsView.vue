<template>
  <div>
    <h1>Running Models</h1>
     <RunningModelsList :runningModels="runningModels"  />
  </div>
</template>

<script>
import axios from 'axios';
import RunningModelsList from '@/components/RunningModelsList.vue'; // Импорт нового компонента

const API_BASE_URL = 'http://localhost:8000';

export default {
  components: {
    RunningModelsList, // Регистрация компонента

  },
  data() {
    return {
      runningModels: [],

    };
  },
  methods: {
    async fetchRunningModels() {
      try {
        const response = await axios.get(`${API_BASE_URL}/running_models/`);
        this.runningModels = response.data;
      } catch (error) {
        console.error('Error fetching running models:', error);
      }
    },
  },
  created() {
    this.fetchRunningModels();
  },

};
</script>
