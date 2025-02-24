# frontend/Dockerfile
# Базовый образ Node.js (LTS версия)
FROM node:lts-alpine

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем package.json и package-lock.json (если есть)
COPY package*.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем остальные файлы проекта
COPY . .

# Собираем проект (если нужно)
RUN npm run build

# Устанавливаем serve (для статического обслуживания)
RUN npm install -g serve

# Запуск приложения (serve -s dist -l 80)
CMD ["serve", "-s", "dist", "-l", "80"]

#EXPOSE 80  # Не обязательно, если используешь serve (и порты в docker-compose.yml)

