import logging
from pyspark.sql import SparkSession


class SanityChecker:
    """Класс для проверки работоспособности платформы Spark."""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_wordcount(self) -> None:
        self.logger.info("Запуск тестового задания WordCount...")
        data = ["Spark is awesome", "Hello Spark", "PySpark makes big data easy"]
        rdd = self.spark.sparkContext.parallelize(data)

        counts = rdd.flatMap(lambda line: line.split(" ")) \
            .map(lambda word: (word, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .collect()

        self.logger.info(f"Результат WordCount: {counts}")