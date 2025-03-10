import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue';
import TrainedModelsView from '../views/TrainedModelsView.vue';
import RunningModelsView from '../views/RunningModelsView.vue';

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
  // Добавьте другие маршруты по мере необходимости
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;