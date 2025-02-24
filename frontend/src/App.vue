<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div id="app">
    <nav>
      <!--  Навигация (пока не нужна, но можно добавить в будущем) -->
    </nav>
    <router-view />  <!-- Здесь будут отображаться компоненты в зависимости от роута -->
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
/* Общие стили приложения */
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
