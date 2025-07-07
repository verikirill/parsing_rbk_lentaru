FROM python:3.10-slim-buster

# Установка необходимых системных пакетов для сборки
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Настройка pip для решения проблем с SSL и сетью
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org"

# Копируем только requirements.txt сначала для кэширования слоя с зависимостями
COPY requirements.txt ./

# Устанавливаем зависимости с дополнительными настройками
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Создаем пользователя без прав root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 