<template>
  <Toast />
  <div id="app">
    <Menubar :model="menuItems" class="custom-menubar">
      <template #start>
        <span class="app-title">ML Platform</span>
      </template>
    </Menubar>
    <div class="content-container">
      <router-view />
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import Menubar from 'primevue/menubar'

const router = useRouter()
const menuItems = ref([
  { label: 'Home', icon: 'pi pi-home', command: () => router.push('/') },
  { label: 'Trained Models', icon: 'pi pi-database', command: () => router.push('/trained-models') },
  { label: 'Running Models', icon: 'pi pi-chart-line', command: () => router.push('/running-models') }
])

</script>

<style>
/* Общие стили приложения (с градиентом) */
#app {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Более современный шрифт */
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  /*color: #ffffff; /* Белый текст для контраста */
  min-height: 100vh; /* Занимаем всю высоту экрана */
  /*background: linear-gradient(135deg, #29323c 0%, #485563 100%); /* Темно-синий градиент фона */
  display: flex;
  flex-direction: column;
}

.content-container {
    padding: 30px; /* Больше отступов */
    flex-grow: 1; /* Растягиваем контент на всю доступную высоту */
    /*  Можно добавить легкий фон, если контент светлый, чтобы улучшить читаемость
    background-color: rgba(255, 255, 255, 0.1);  Слегка прозрачный белый фон
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    */
}

.custom-menubar {
  background: linear-gradient(135deg, #4A148C 0%, #673AB7 100%); /* Фиолетовый градиент */
  border-radius: 0;
  border: none;
  padding: 0.5rem 1rem; /* Немного уменьшим паддинги */
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); /* Более заметная тень */
  width: 100%;  /*На всю ширину*/
  display: flex; /* Используем flexbox */
  align-items: center; /* Выравниваем элементы по центру по вертикали */
}

.custom-menubar .p-menubar-root-list {
    margin-left: 2rem; /* Отступ слева для списка пунктов меню */
}

.custom-menubar .p-menuitem .p-menuitem-link .p-menuitem-text {
  color: #eee; /* Светло-серый цвет текста (почти белый) */
    transition: background-color 0.2s ease-in-out;
    border-radius: 5px;
}
.custom-menubar .p-menubar-root-list > .p-menuitem > .p-menuitem-content:hover {
    background-color: rgba(255, 255, 255, 0.253); /* Эффект при наведении */
}

.custom-menubar .p-menuitem .p-menuitem-link .p-menuitem-icon {
  color: #eee; /* Светло-серый для иконок */
    margin-right: 0.5rem; /* Добавляем небольшой отступ между иконкой и текстом */
}
/* Стили для активного пункта меню */
.custom-menubar .p-menuitem.p-menuitem-active > .p-menuitem-content > .p-menuitem-link {
  background-color: rgba(255, 255, 255, 0.2) !important;
  color: #ffffff !important;
}

/* Стили для наведения */
.custom-menubar .p-menubar-root-list > .p-focus > .p-menuitem-content:hover {
  background-color: rgba(255, 255, 255, 0.25) !important;
  color: #fff !important;
}

/* Остальные стили остаются без изменений */
.custom-menubar .p-menuitem-active > .p-menuitem-content > .p-menuitem-link,
.custom-menubar .p-menuitem-active > .p-menuitem-content > .p-menuitem-link:hover {
  background-color: rgba(255, 255, 255, 0.2); /* Цвет фона для активного пункта */
  color: #ffffff; /* Цвет текста для активного пункта */
}

/* Стили для заголовка приложения */
.app-title {
    font-size: 1.8rem; /* Еще больше размер шрифта */
    font-weight: bold;
    color: #ffffff; /* Белый цвет для заголовка */
    margin-left: 1rem;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* Тень для текста */
}
</style>
