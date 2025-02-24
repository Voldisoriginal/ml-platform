<template>
  <div>
      <h2>Train Settings</h2>
      <div>
        <label for="target-column">Target Column:</label>
        <select id="target-column" v-model="selectedTargetColumn" required>
          <option v-for="column in columns" :key="column" :value="column">{{ column }}</option>
        </select>
      </div>

      <div>
          <label for="train-size">Train Size:</label>
          <input type="number" id="train-size" v-model.number="trainSize" min="0.1" max="0.9" step="0.1" required>
      </div>
      <div>
          <label for="random-state">Random State:</label>
          <input type="number" id="random-state" v-model.number="randomState" required>
      </div>

      <button @click="submitSettings">Confirm Settings</button>

  </div>
</template>

<script>

export default {
name: 'TrainSettings',
  props:{
    columns: {
        type: Array,
        required: true,
    }
  },
data() {
  return {
      selectedTargetColumn: null,
    trainSize: 0.7,
    randomState: 42
  };
},
  watch: {
  // очень важный момент - каждый раз как меняются колонки - target обнуляется.
      columns(newColumns) {
          if (newColumns && newColumns.length > 0){
              this.selectedTargetColumn = newColumns[0]; // Выбираем первый столбец по умолчанию
          } else {
            this.selectedTargetColumn = null;
          }

      }
  },
  mounted() {
    // Установка начальных данных
      if (this.columns && this.columns.length > 0){
         this.selectedTargetColumn = this.columns[0];
      }
  },
methods: {
  submitSettings() {
    if (!this.selectedTargetColumn) {
        alert('Please select a target column.');
        return;
    }
    this.$emit('settings-submitted', {
        targetColumn: this.selectedTargetColumn,
      trainSize: this.trainSize,
      randomState: this.randomState
    });
  }
}
};
</script>
