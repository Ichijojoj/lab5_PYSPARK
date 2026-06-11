FROM apache/spark:3.5.1-scala2.12-java11-ubuntu

USER root

# Установка системных зависимостей для сборки Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочих директорий
WORKDIR /app

# Копирование и установка зависимостей Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Скачивание и добавление Oracle JDBC-драйвера в Spark jars
ADD https://repo1.maven.org/maven2/com/oracle/database/jdbc/ojdbc8/21.1.0.0/ojdbc8-21.1.0.0.jar /opt/spark/jars/ojdbc8.jar
RUN chmod 644 /opt/spark/jars/ojdbc8.jar

# Копирование исходного кода приложения
COPY src/ /app/src/
COPY main.py /app/

# Назначение прав пользователю spark во избежание проблем безопасности в k8s (securityContext)
RUN chown -R 185:185 /app
USER 185

ENV PYTHONPATH="/app"

ENTRYPOINT ["/opt/spark/bin/spark-submit", "/app/main.py"]