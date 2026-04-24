import logging
from pyspark.sql import SparkSession


class SparkManager:

    def __init__(self, app_name: str):
        self.app_name = app_name
        self._spark = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def spark(self) -> SparkSession:
        if self._spark is None:
            self.logger.info(f"Инициализация Spark сессии: {self.app_name}")
            import findspark
            findspark.init()

            self._spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
            self._spark.sparkContext.setLogLevel("ERROR")
        return self._spark

    def stop(self) -> None:
        if self._spark:
            self.logger.info("Завершение Spark сессии.")
            self._spark.stop()