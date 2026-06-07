from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SparkConfig:
    app_name: str = "Oracle_KMeans_Clustering_Mart"
    master: str = "local[*]"

    configs: Dict[str, str] = field(default_factory=lambda: {
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",

        # 1. Оптимизация сериализации данных
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "512m",

        # 2. Оптимизация интеграции Python и JVM через Apache Arrow
        "spark.sql.execution.arrow.pyspark.enabled": "true",

        # 3. Пути к драйверам баз данных (включая Oracle JDBC)
        "spark.jars": "/opt/spark/jars/ojdbc8.jar",
        "spark.driver.extraClassPath": "/opt/spark/jars/ojdbc8.jar",
        "spark.executor.extraClassPath": "/opt/spark/jars/ojdbc8.jar",

        # 4. Тюнинг параллелизма и разделов (подгоняется под количество ядер процессора локальной машины)
        "spark.sql.shuffle.partitions": "8",
        "spark.default.parallelism": "8",

        # 5. Очистка локальных метаданных и временных файлов для предотвращения утечки памяти
        "spark.cleaner.periodicGC.interval": "10min"
    })