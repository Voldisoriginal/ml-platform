import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue';
import TrainedModelsView from '../views/TrainedModelsView.vue';
import RunningModelsView from '../views/RunningModelsView.vue';
import DatasetsPage from '../views/DatasetsPage.vue'; // Import your DatasetsPage

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomeView
  },
  {
    path: '/trained-models',
    name: 'TrainedModels',
    component: TrainedModelsView
  },
  {
    path: '/running-models',
    name: 'RunningModels',
    component: RunningModelsView
  },
  {
    path: '/datasets', // The path for your datasets page
    name: 'Datasets',
    component: DatasetsPage, // Link to your DatasetsPage component
  },
  // Добавьте другие маршруты по мере необходимости
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;