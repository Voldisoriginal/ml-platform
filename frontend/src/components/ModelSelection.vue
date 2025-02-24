<template>
  <div>
    <h2>Select Model</h2>
    <div v-for="model in models" :key="model.type">
      <input
        type="radio"
        :id="model.type"
        :value="model.type"
        v-model="selectedModelType"
        @change="selectModel(model)"
      />
      <label :for="model.type">{{ model.name }}</label>

      <!-- Параметры для DecisionTreeRegressor -->
      <div v-if="model.type === 'DecisionTreeRegressor' && selectedModelType === 'DecisionTreeRegressor'">
        <label for="max-depth">Max Depth:</label>
        <input type="number" id="max-depth" v-model.number="model.params.max_depth" />

        <label for="min-samples-split">Min Samples Split:</label>
        <input type="number" id="min-samples-split" v-model.number="model.params.min_samples_split" />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ModelSelection',
  props: {
    models: {
      type: Array,
      required: true
    }
  },
  data() {
    return {
      selectedModelType: null,
    };
  },
  methods: {
    selectModel(model) {
        // Важно создать копию объекта параметров,
        // чтобы изменения в этом компоненте не влияли напрямую на объект в родительском компоненте.
        const selectedModel = {
            model_type: model.type,
            params: { ...model.params }  // Копируем параметры
        };
        this.$emit('model-selected', selectedModel);

    }
  }
};
</script>
