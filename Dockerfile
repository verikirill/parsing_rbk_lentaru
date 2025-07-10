FROM python:3.10-slim-buster

# Установка необходимых системных пакетов для сборки
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя раньше, чтобы настроить правильные права
RUN useradd -m appuser

WORKDIR /app

# Настройка pip для решения проблем с SSL и сетью
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org" \
    TRANSFORMERS_CACHE=/home/appuser/.cache/transformers \
    HF_HOME=/home/appuser/.cache/huggingface

# Создаем папки для кэша и даем права пользователю
RUN mkdir -p /home/appuser/.cache/transformers \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache

# Копируем только requirements.txt сначала для кэширования слоя с зависимостями
COPY requirements.txt ./

# Устанавливаем зависимости с дополнительными настройками
RUN pip install --upgrade pip && \
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Даем права пользователю на рабочую папку
RUN chown -R appuser:appuser /app

# Переключаемся на пользователя без прав root
USER appuser

EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 