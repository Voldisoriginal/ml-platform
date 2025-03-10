<template>
  <div>
    <h2>Trained Models</h2>

    <!-- Search and Filter Inputs -->
    <div class="filters">
      <div>
        <label for="search">Search:</label>
        <input type="text" id="search" v-model="searchQuery" placeholder="Search by ID, dataset, or target">
      </div>

      <div>
        <label for="model-type">Model Type:</label>
        <select id="model-type" v-model="selectedModelType">
          <option value="">All</option>
          <option value="LinearRegression">Linear Regression</option>
          <option value="DecisionTreeRegressor">Decision Tree Regressor</option>
          <!-- Add other model types here -->
        </select>
      </div>

      <div>
        <label for="dataset-filter"> Dataset:</label>
        <input type="text" id="dataset-filter" v-model="selectedDatasetFilename" placeholder="Filter by dataset">
      </div>

      <div>
        <label for="sort-by">Sort By:</label>
        <select id="sort-by" v-model="selectedSortBy">
          <option value="start_time">Start Time</option>
          <option value="r2_score">R2 Score</option>
          <option value="mse">MSE</option>
          <!-- Add other sort options here -->
        </select>
      </div>
      <div>
        <label for="sort-order">Sort Order:</label>
        <select id="sort-order" v-model="selectedSortOrder">
          <option value="desc">Descending</option>
          <option value="asc">Ascending</option>
        </select>
      </div>
    </div>
    <div v-if="models.length === 0">No trained models yet.</div>
    <table v-else>
        <thead>
          <tr>
            <th>Model ID</th>
            <th>Dataset</th>
            <th>Target Column</th>
            <th>Model Type</th>
            <th>Metrics</th>
            <th>Actions</th>
          </tr>
        </thead>
      <tbody>
          <tr v-for="model in filteredModels" :key="model.id" @click="showModelDetails(model)">
            <td>{{ model.id }}</td>
            <td>{{ formatFilename(model.dataset_filename) }}</td>
            <td>{{ model.target_column }}</td>
            <td>{{ model.model_type }}</td>
            <td>
              <ul>
                <li v-for="(value, key) in model.metrics" :key="key">
                  {{ key }}: {{ value.toFixed(4) }}
                </li>
              </ul>
            </td>
              <td>
                  <button @click.stop="selectForInference(model.id)">Select for Inference</button>
              </td>
          </tr>
        </tbody>
      </table>
  </div>
</template>

<script>
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default {
  props: {
    models: {
      type: Array,
      required: true,
    },
  },
  data() {
    return {
      searchQuery: '',
      selectedSortBy: 'start_time',
      selectedSortOrder: 'desc',
      selectedModelType: '',
      selectedDatasetFilename: '',
    };
  },
  methods: {
    selectForInference(modelId) {
      this.$emit('model-selected-for-inference', modelId);
    },
    getDatasetDownloadLink(filename) {
      return `${API_BASE_URL}/download_dataset/${filename}`;
    },
    formatFilename(filename) {
      if (!filename) {
        return '';
      }
      const parts = filename.split('_');
      if (parts.length > 1) {
        return parts.slice(1).join('_');
      }
      return filename;
    },
      showModelDetails(model) {
          this.$emit('show-model-details', model); // Emit event to parent
      },
  },
    computed: {
    filteredModels() {
      // Apply filtering and sorting logic here, if needed, otherwise return this.models
      let filtered = [...this.models];

       if (this.searchQuery) {
            const query = this.searchQuery.toLowerCase();
            filtered = filtered.filter(model =>
                model.id.toLowerCase().includes(query) ||
                (model.dataset_filename && model.dataset_filename.toLowerCase().includes(query)) ||
                (model.target_column && model.target_column.toLowerCase().includes(query))
            );
        }


        // Sorting logic
        if (this.selectedSortBy) {
            filtered.sort((a, b) => {
                const order = this.selectedSortOrder === 'asc' ? 1 : -1;
                 if (this.selectedSortBy === 'r2_score' || this.selectedSortBy === 'mse') {
                    // Special handling for metrics, check they exist
                    const valA = a.metrics && a.metrics[this.selectedSortBy] !== undefined ? a.metrics[this.selectedSortBy] : (this.selectedSortOrder === 'asc' ? -Infinity : Infinity);
                    const valB = b.metrics && b.metrics[this.selectedSortBy] !== undefined ? b.metrics[this.selectedSortBy] : (this.selectedSortOrder === 'asc' ? -Infinity : Infinity);

                    return (valA - valB) * order;
                }
                else if (this.selectedSortBy === 'start_time')
                {
                    return (new Date(a.start_time) - new Date(b.start_time)) * order;
                }
                else {
                    // Generic string comparison, check that values are defined
                    const valA = a[this.selectedSortBy] ? a[this.selectedSortBy].toString().toLowerCase() : '';
                    const valB = b[this.selectedSortBy] ? b[this.selectedSortBy].toString().toLowerCase() : '';                    return valA.localeCompare(valB) * order;
                }

            });
        }



      return filtered;
    },
  },
};
</script>

<style scoped>
/* Filters */
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
  justify-content: center;
}

.filters > div {
  /*margin: 5px 5px 5px 5px;*/
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

th,
td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

th {
  background-color: #f2f2f2;
}
tr{
    cursor: pointer;
}

</style>
