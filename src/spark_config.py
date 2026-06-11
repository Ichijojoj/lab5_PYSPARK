import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SparkConfig:
    """Конфигурация Spark, оптимизированная под запуск в контейнере Kubernetes."""

    app_name: str = "K8s_PySpark_KMeans_Clustering"

    master: str = os.getenv("SPARK_MASTER", "local[*]")

    configs: Dict[str, str] = field(default_factory=lambda: {
        # Лимиты ресурсов на контейнер (синхронизировано с k8s limits)
        "spark.driver.memory": os.getenv("SPARK_DRIVER_MEMORY", "2g"),
        "spark.executor.memory": os.getenv("SPARK_EXECUTOR_MEMORY", "2g"),
        "spark.kubernetes.container.image": os.getenv("SPARK_IMAGE", "pyspark-k8s-app:latest"),

        # Оптимизация сериализации данных
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "512m",

        # Интеграция Python и JVM через Apache Arrow
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # Пути к JDBC-драйверам внутри k8s-контейнера
        "spark.jars": "/opt/spark/jars/ojdbc8.jar",
        "spark.driver.extraClassPath": "/opt/spark/jars/ojdbc8.jar",
        "spark.executor.extraClassPath": "/opt/spark/jars/ojdbc8.jar",

        # Тюнинг параллелизма в зависимости от ядер подов
        "spark.sql.shuffle.partitions": os.getenv("SPARK_SHUFFLE_PARTITIONS", "4"),
        "spark.default.parallelism": os.getenv("SPARK_DEFAULT_PARALLELISM", "4"),

        # Очистка ресурсов во избежание утечек памяти в долгоживущих подах
        "spark.cleaner.periodicGC.interval": "15min"
    })