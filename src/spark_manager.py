import logging
from typing import Dict
from pyspark.sql import SparkSession


class SparkManager:

    def __init__(self, spark_configs: Dict[str, str]):
        self.spark_configs = spark_configs
        self._spark = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            app_name = self.spark_configs.get("spark.app.name", "Default_App")
            self.logger.info(f"Инициализация Spark-сессии: {app_name}")

            try:
                import findspark
                findspark.init()
            except ImportError as e:
                self.logger.warning(
                    f"Библиотека findspark не импортирована: {e}. "
                    "Инициализация продолжится в стандартном системном окружении."
                )

            builder = SparkSession.builder
            for key, value in self.spark_configs.items():
                builder = builder.config(key, value)

            self._spark = builder.getOrCreate()
            self._spark.sparkContext.setLogLevel("ERROR")
        return self._spark

    def stop(self) -> None:
        if self._spark:
            self.logger.info("Завершение Spark-сессии.")
            self._spark.stop()