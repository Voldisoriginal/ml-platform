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
const app = createApp(App)
app.use(router)
app.use(PrimeVue)
app.use(ToastService)
app.mount('#app')