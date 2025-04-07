<template>
  <div class="model-selection-card p-card">
    <div class="p-card-title">
      <h2>Select Model</h2>
    </div>
    <div class="p-card-content">
      <TabView v-model:activeIndex="activeTabIndex">
        <TabPanel header="Regression">
          <div v-if="regressionModels.length === 0">
            No regression models available.
          </div>
          <div v-else class="model-list">
            <div v-for="model in regressionModels" :key="model.type" class="p-field-radiobutton model-item">
              <RadioButton
                :inputId="model.type"
                name="regressionModel"
                :value="model.type"
                v-model="selectedModelTypeInternal"
                @change="handleModelSelection(model)"
              />
              <label :for="model.type" class="p-ml-2">{{ model.name }}</label>
            </div>
          </div>
        </TabPanel>
        <TabPanel header="Classification">
           <div v-if="classificationModels.length === 0">
             No classification models available.
           </div>
           <div v-else class="model-list">
             <div v-for="model in classificationModels" :key="model.type" class="p-field-radiobutton model-item">
               <RadioButton
                 :inputId="model.type"
                 name="classificationModel"
                 :value="model.type"
                 v-model="selectedModelTypeInternal"
                 @change="handleModelSelection(model)"
               />
               <label :for="model.type" class="p-ml-2">{{ model.name }}</label>
             </div>
           </div>
        </TabPanel>
      </TabView>

      <!-- Динамические поля для параметров выбранной модели -->
      <div v-if="selectedModelDefinition && selectedModelDefinition.parameters.length > 0" class="p-mt-4 model-parameters">
        <h4>Parameters for {{ selectedModelDefinition.name }}</h4>
        <div class="p-grid p-formgrid p-fluid">
          <div v-for="param in selectedModelDefinition.parameters" :key="param.name" class="p-field p-col-12 p-md-6">
             <label :for="param.name">{{ param.label }}</label>
             <!-- Используем v-if для выбора компонента PrimeVue -->
              <InputNumber
              v-if="param.component === 'InputNumber'"
              :inputId="param.name"
              v-model="currentParams[param.name]"
              mode="decimal"
              :minFractionDigits="param.type === 'integer' ? 0 : (param.validation?.minFractionDigits ?? 0)" 
              :maxFractionDigits="param.type === 'integer' ? 0 : (param.validation?.maxFractionDigits ?? 5)"
              :min="param.validation?.min"
              :max="param.validation?.max"
              :step="param.type === 'float' ? (param.validation?.step ?? 0.1) : (param.validation?.step ?? 1)"
              showButtons
              :allowEmpty="true"
              @update:modelValue="emitSelection"
              />
             <Dropdown
                v-else-if="param.component === 'Dropdown'"
                :inputId="param.name"
                v-model="currentParams[param.name]"
                :options="param.validation?.values"
                placeholder="Select..."
                 @change="emitSelection"
              />
             <Checkbox
                v-else-if="param.component === 'Checkbox'"
                :inputId="param.name"
                v-model="currentParams[param.name]"
                :binary="true"
                 @change="emitSelection"
              />
             <!-- Добавить InputText или другие компоненты при необходимости -->
              <InputText
                v-else-if="param.component === 'InputText'"
                :inputId="param.name"
                v-model="currentParams[param.name]"
                 @update:modelValue="emitSelection"
               />
             <small v-if="param.validation?.min !== undefined || param.validation?.max !== undefined" class="p-d-block p-mt-1">
                {{ param.validation?.min !== undefined ? `Min: ${param.validation.min}` : '' }}
                {{ param.validation?.max !== undefined ? ` Max: ${param.validation.max}` : '' }}
             </small>
          </div>
        </div>
      </div>
      <div v-else-if="selectedModelDefinition && selectedModelDefinition.parameters.length === 0" class="p-mt-4">
         <p>No configurable parameters for {{ selectedModelDefinition.name }}.</p>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, reactive, defineProps, defineEmits } from 'vue';
import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';
import RadioButton from 'primevue/radiobutton';
import InputNumber from 'primevue/inputnumber';
import Dropdown from 'primevue/dropdown';
import Checkbox from 'primevue/checkbox';
import InputText from 'primevue/inputtext'; // Добавили InputText

const props = defineProps({
  models: { // Теперь это список объектов AvailableModelInfo с бэкенда
    type: Array,
    required: true,
    default: () => [] // Безопасное значение по умолчанию
  },
  // Можно передать тип задачи извне, если он определен заранее (например, по таргету)
  // targetTaskType: {
  //   type: String, // 'regression' or 'classification' or null
  //   default: null
  // }
});

const emit = defineEmits(['model-selected', 'update:selectedModelType']); // Добавим второй эмит для v-model

const activeTabIndex = ref(0); // 0 для регрессии, 1 для классификации
const selectedModelTypeInternal = ref(null); // Внутренний стейт для выбранного типа модели
const currentParams = reactive({}); // Реактивный объект для хранения текущих значений параметров

// --- Computed Properties ---

// Фильтруем модели по типу задачи
const regressionModels = computed(() => {
  return props.models.filter(m => m.task_type === 'regression');
});

const classificationModels = computed(() => {
  return props.models.filter(m => m.task_type === 'classification');
});

// Находим полное определение выбранной модели
const selectedModelDefinition = computed(() => {
  if (!selectedModelTypeInternal.value) return null;
  return props.models.find(m => m.type === selectedModelTypeInternal.value);
});

// --- Watchers ---

// Следим за изменением списка доступных моделей (если он может измениться динамически)
watch(() => props.models, (newModels) => {
  // Сбросить выбор, если текущая выбранная модель больше не доступна
  if (selectedModelTypeInternal.value && !newModels.some(m => m.type === selectedModelTypeInternal.value)) {
     resetSelection();
  }
  // Опционально: автоматически выбрать первую модель в активной вкладке
  // if (!selectedModelTypeInternal.value) {
  //    const currentTabModels = activeTabIndex.value === 0 ? regressionModels.value : classificationModels.value;
  //    if (currentTabModels.length > 0) {
  //        handleModelSelection(currentTabModels[0]);
  //    }
  // }
}, { deep: true });


// --- Methods ---

// Срабатывает при клике на RadioButton
const handleModelSelection = (modelDefinition) => {
  if (!modelDefinition) {
     resetSelection();
     return;
  }
  console.log("Selected Model Definition:", modelDefinition);
  selectedModelTypeInternal.value = modelDefinition.type;

  // Очищаем старые параметры
  Object.keys(currentParams).forEach(key => delete currentParams[key]);

  // Устанавливаем параметры по умолчанию для новой модели
  if (modelDefinition.parameters) {
    modelDefinition.parameters.forEach(param => {
      // Важно: Преобразуем типы для InputNumber, если нужно
      if (param.type === 'integer' || param.type === 'float') {
         currentParams[param.name] = param.default !== null ? Number(param.default) : null;
      } else {
         currentParams[param.name] = param.default;
      }
    });
  }
  console.log("Initialized Params:", JSON.parse(JSON.stringify(currentParams))); // Логируем копию
  emitSelection(); // Сразу отправляем событие после выбора модели и установки параметров по умолчанию
};

// Сброс выбора
const resetSelection = () => {
  selectedModelTypeInternal.value = null;
  Object.keys(currentParams).forEach(key => delete currentParams[key]);
  emitSelection(); // Отправляем null/пустой объект
}

// Отправка события родителю
const emitSelection = () => {
    // Небольшая задержка, чтобы Vue успел обновить currentParams после ввода
    // Это может быть нужно для @update:modelValue у InputText/InputNumber
    setTimeout(() => {
        if (!selectedModelTypeInternal.value) {
            emit('model-selected', null);
            return;
        }

        // Создаем копию параметров для отправки, чтобы избежать прямой мутации
        const paramsToSend = { ...currentParams };

        // Опционально: Приведение типов перед отправкой (на случай если v-model не справился)
        selectedModelDefinition.value?.parameters.forEach(param => {
            if (paramsToSend[param.name] !== null && paramsToSend[param.name] !== undefined) {
                if (param.type === 'integer') {
                    paramsToSend[param.name] = parseInt(paramsToSend[param.name], 10);
                } else if (param.type === 'float') {
                    paramsToSend[param.name] = parseFloat(paramsToSend[param.name]);
                }
                // Удаляем параметры, если они null или пустые строки (опционально, зависит от бэкенда)
                // if (paramsToSend[param.name] === null || paramsToSend[param.name] === '') {
                //     delete paramsToSend[param.name];
                // }
            } else {
                 // Если значение null/undefined, удаляем его, чтобы бэкенд использовал свои дефолты, если они есть
                 // Или отправляем null, если бэкенд этого ожидает
                 // delete paramsToSend[param.name];
            }
        });


        const selection = {
            model_type: selectedModelTypeInternal.value,
            params: paramsToSend
        };
        console.log("Emitting model-selected:", JSON.parse(JSON.stringify(selection)));
        emit('model-selected', selection);
   }, 0); // Задержка 0 мс достаточно для следующего тика
};

// Можно добавить watch для activeTabIndex, чтобы сбрасывать выбор при смене таба
watch(activeTabIndex, () => {
    // При смене таба, ищем, выбрана ли модель из этого таба
    const currentTabModels = activeTabIndex.value === 0 ? regressionModels.value : classificationModels.value;
    const isSelectedModelInCurrentTab = currentTabModels.some(m => m.type === selectedModelTypeInternal.value);

    // Если выбранная модель не из текущего таба, сбрасываем выбор
    if (!isSelectedModelInCurrentTab) {
         resetSelection();
    }
});

</script>

<style scoped>
.model-selection-card {
  margin-top: 2rem; /* Отступ сверху */
  margin-bottom: 2rem; /* Отступ снизу */
}

.model-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem; /* Пространство между моделями */
  padding: 1rem 0; /* Небольшие отступы сверху/снизу внутри списка */
}

.model-item {
  /* Стили для элемента списка моделей, если нужно */
   margin-bottom: 0; /* Убираем лишний отступ у p-field */
}

.model-parameters {
  border-top: 1px solid var(--surface-d); /* Разделитель */
  padding-top: 1.5rem; /* Отступ сверху для секции параметров */
  margin-top: 1.5rem; /* Отступ сверху от списка моделей */
}

.model-parameters h4 {
  margin-bottom: 1rem; /* Отступ под заголовком параметров */
  color: var(--primary-color); /* Цвет заголовка */
}

/* Улучшение внешнего вида полей формы */
.p-field > label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: bold;
}

/* Для сетки параметров */
.p-formgrid.p-grid {
  margin-top: 1rem;
}

.p-field.p-col-12.p-md-6 {
    padding: 0.5rem; /* Добавим немного воздуха вокруг полей */
}

/* Стилизация TabView */
:deep(.p-tabview-nav) {
  background-color: var(--surface-b); /* Фон вкладок */
}

:deep(.p-tabview-nav-link) {
    background-color: var(--surface-c) !important; /* Фон неактивной вкладки */
    border: 1px solid var(--surface-d) !important;
    border-bottom: 0 !important;
    color: var(--text-color-secondary) !important;
    margin-right: 2px;
}

:deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link) {
    background-color: var(--surface-a) !important; /* Фон активной вкладки */
    border-color: var(--surface-d) !important;
    color: var(--primary-color) !important; /* Цвет текста активной вкладки */
}

:deep(.p-tabview-panels) {
    background-color: var(--surface-a); /* Фон панели контента */
     padding: 1rem;
     border: 1px solid var(--surface-d);
     border-top: 0; /* Убираем верхнюю границу, т.к. она есть у вкладок */
}

/* Адаптивность */
@media (max-width: 768px) {
  .p-field.p-col-12.p-md-6 {
    width: 100%; /* Поля занимают всю ширину на маленьких экранах */
    padding: 0.5rem 0;
  }
}

</style>