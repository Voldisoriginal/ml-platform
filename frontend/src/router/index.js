import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue';
import TrainedModelsView from '../views/TrainedModelsView.vue';
import RunningModelsView from '../views/RunningModelsView.vue';
import DatasetsPage from '../views/DatasetsPage.vue'; // Import your DatasetsPage
import InferencePage from '../views/InferencePage.vue'; 
import DatasetDetailView from '../views/DatasetDetailView.vue';

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
  {
    path: '/inference/:modelId', // Dynamic segment for model ID
    name: 'inference',
    component: InferencePage,
    props: true // Automatically pass route params as props
  },
  { // New Route for Dataset Details
    path: '/datasets/:datasetId', // Use a dynamic segment for the ID
    name: 'DatasetDetail',
    component: DatasetDetailView,
    props: true // Automatically pass route params as props to the component
  },
  // Добавьте другие маршруты по мере необходимости
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;