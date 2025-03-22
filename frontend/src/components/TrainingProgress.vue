<template>
  <Card class="training-progress">
    <template #title>
      <h2>Training Progress</h2>
    </template>
    <template #content>
      <div v-if="isTraining">
        <ProgressSpinner v-if="isTraining" strokeWidth="4" animationDuration=".5s" />
        <p class="status-message">Training in progress...</p>
        <p>Task ID: <span class="task-id">{{ taskId }}</span></p>
        <p>Status: <span :class="statusClass">{{ taskStatus }}</span></p>

        <Message v-if="error" severity="error" :closable="false">
          <template #icon>
              <span class="pi pi-exclamation-triangle" style="font-size: 1.5rem"></span>
          </template>
          <h3 class="error-title">Error Details:</h3>
          <p>Type: {{ error.exc_type }}</p>
          <p>Message: {{ error.exc_message }}</p>
           <pre v-if="error.exc_traceback" class="traceback">{{ error.exc_traceback }}</pre>
        </Message>
      </div>

      <div v-else-if="result">
        <h3 class="results-title">Training Results:</h3>
        <p>Model ID: <span class="model-id">{{ result.id }}</span></p>
        <p>Model Type: <span class="model-type">{{ result.model_type }}</span></p>
        <h4>Metrics:</h4>
         <DataTable :value="formatMetrics" responsiveLayout="scroll" :stripedRows="true">
            <Column field="name" header="Metric"></Column>
            <Column field="value" header="Value"></Column>
        </DataTable>
      </div>

      <Message v-else-if="error && !isTraining" severity="error" :closable="false">
          <template #icon>
             <span class="pi pi-times-circle" style="font-size: 1.5rem"></span>
          </template>
          <h3 class="error-title">Error Details:</h3>
          <p>Type: {{ error.exc_type }}</p>
          <p>Message: {{ error.exc_message }}</p>
        <pre v-if="error.exc_traceback"  class="traceback">{{ error.exc_traceback }}</pre>
      </Message>
    </template>
  </Card>
</template>

<script>
import axios from 'axios';
import Card from 'primevue/card';
import ProgressSpinner from 'primevue/progressspinner';
import Message from 'primevue/message';
import DataTable from 'primevue/datatable';  // Import DataTable
import Column from 'primevue/column';        // Import Column

const API_BASE_URL = 'http://localhost:8000';

export default {
  name: 'TrainingProgress',
  components: {
    Card,
    ProgressSpinner,
    Message,
    DataTable,
    Column,
  },
  props: {
    isTraining: {
      type: Boolean,
      required: true,
    },
    result: {
      type: Object,
      required: false,
    },
    taskId: {
      type: String,
      required: false,
    },
  },
  data() {
    return {
      taskStatus: null,
      pollInterval: null,
      error: null,
    };
  },
  computed: {
    statusClass() {
      return {
        'status-pending': this.taskStatus === 'PENDING',
        'status-started': this.taskStatus === 'STARTED',
        'status-success': this.taskStatus === 'SUCCESS',
        'status-failure': this.taskStatus === 'FAILURE',
      };
    },
      formatMetrics() {
          if (!this.result || !this.result.metrics) {
              return [];
          }
        //Convert to array for DataTable
          return Object.entries(this.result.metrics).map(([key, value]) => ({
              name: key,
              value: value
          }));
      }

  },
  watch: {
    taskId(newTaskId) {
      if (newTaskId) {
        this.startPolling();
      } else {
        this.stopPolling();
        this.taskStatus = null;
        this.error = null;
      }
    },
    isTraining(newIsTraining) { // Add watcher for isTraining
        if (!newIsTraining && this.pollInterval) {
            //If isTraining changed false and polling worked => stop polling.
            this.stopPolling();
        }
    }
  },
  methods: {
    async checkTaskStatus() {
      try {
        const response = await axios.get(`${API_BASE_URL}/train_status/${this.taskId}`);
        this.taskStatus = response.data.status;

        if (response.data.status === 'SUCCESS') {
          this.$emit('training-complete', response.data.result);  // Keep this emit
          this.stopPolling();
          // Emit to parent component.
          this.$emit('update:isTraining', false); // Use update:propName for two-way binding
          this.$emit('update:result', response.data.result);  //  and results
          this.error = null;

        } else if (response.data.status === 'FAILURE') {
          this.error = response.data.error;
          console.error("Training failed:", this.error);
        } else {
          this.error = null;
        }

      } catch (error) {
        console.error("Error checking task status:", error);
        this.error = { exc_type: "NetworkError", exc_message: "Failed to fetch status." };
      }
    },
    startPolling() {
      this.stopPolling();
      this.pollInterval = setInterval(this.checkTaskStatus, 2000);
    },
    stopPolling() {
      if (this.pollInterval) {
        clearInterval(this.pollInterval);
        this.pollInterval = null;
      }
    },
  },
  beforeUnmount() {
    this.stopPolling();
  },
};
</script>

<style scoped>
.training-progress {
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  background-color: #fff;
  text-align: center;  /* Center content */
}

.status-message {
  margin-top: 1rem;
  margin-bottom: 1rem;
  font-size: 1.2em; /* Larger font for status */
}

.task-id {
    font-family: monospace;
    color: #555;
}
.model-id, .model-type{
    font-style: italic;
}

.status-pending {
  color: #ff9800; /* Orange for pending */
}

.status-started {
  color: #2196f3; /* Blue for started */
}

.status-success {
  color: #4caf50; /* Green for success */
}

.status-failure {
  color: #f44336; /* Red for failure */
}

.error-title {
    margin-top: 0.5rem;
  margin-bottom: 0.5rem;
  color: #d32f2f;
}
.results-title{
    margin-bottom: 1rem;
}
.traceback {
  white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
  text-align: left;
  font-family: monospace;
  background-color: #f8f8f8;
  border: 1px solid #ddd;
    padding: 10px;
    overflow: auto;
}
</style>
