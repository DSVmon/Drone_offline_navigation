#!/bin/bash

# Скрипт для автоматического запуска симуляции автономного спасательного дрона (Phase 4 MVP)

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}[INFO] Запуск системы симуляции спасательного дрона...${NC}"

# 1. Проверка зависимостей
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker не установлен. Пожалуйста, установите Docker перед запуском.${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Compose не найден. Убедитесь, что установлен плагин docker-compose-v2.${NC}"
    exit 1
fi

# 2. Настройка прав для GUI (X11)
echo -e "${GREEN}[INFO] Настройка прав доступа к X11...${NC}"
if command -v xhost &> /dev/null; then
    xhost +local:docker > /dev/null
else
    echo -e "${RED}[WARN] Утилита xhost не найдена. GUI может не запуститься, если не настроены права доступа к X11.${NC}"
fi

# Функция очистки при выходе
cleanup() {
    echo -e "\n${GREEN}[INFO] Остановка контейнеров и очистка прав X11...${NC}"
    docker compose -f docker/docker-compose.yml down
    if command -v xhost &> /dev/null; then
        xhost -local:docker > /dev/null
    fi
    echo -e "${GREEN}[INFO] Завершено.${NC}"
}

# Перехват Ctrl+C
trap cleanup SIGINT

# 3. Запуск Docker Compose
echo -e "${GREEN}[INFO] Сборка и запуск контейнеров...${NC}"
docker compose -f docker/docker-compose.yml up --build

# Вызов очистки после штатного завершения
cleanup
