import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import PrimeVue from 'primevue/config'
//import 'primevue/resources/themes/lara-light-indigo/theme.css'  // Тема
import './assets/styles/primevue-overrides.css'
import './assets/styles/refer.css'
import 'primevue/resources/primevue.min.css'            // Базовые стили
import 'primeicons/primeicons.css'                      // Иконки
import ToastService from 'primevue/toastservice'
import Tooltip from 'primevue/tooltip';

import { Chart, registerables } from 'chart.js';
import { BoxPlotController, BoxAndWiskers } from '@sgratzl/chartjs-chart-boxplot';
import { MatrixController, MatrixElement } from 'chartjs-chart-matrix';

Chart.register(...registerables, BoxPlotController, BoxAndWiskers, MatrixController, MatrixElement);
const app = createApp(App)
app.use(router)
app.use(PrimeVue)
app.use(ToastService)
app.directive('tooltip', Tooltip);
app.mount('#app')