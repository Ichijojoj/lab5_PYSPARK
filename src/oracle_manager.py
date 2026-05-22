import logging
import time
from pyspark.sql import SparkSession, DataFrame
from py4j.protocol import Py4JJavaError


class OracleManager:
    """Управление взаимодействием с БД Oracle (Extract & Load)."""

    def __init__(self, spark: SparkSession, config):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_data(self, retries=15) -> DataFrame:
        """Читает данные из Oracle (Таблицы-витрины)."""
        delay = 10
        for attempt in range(1, retries + 1):
            try:
                self.logger.info(f"Чтение витрины {self.config.source_table} из БД...")
                df = self.spark.read \
                    .format("jdbc") \
                    .option("url", self.config.db_url) \
                    .option("dbtable", self.config.source_table) \
                    .option("user", self.config.db_user) \
                    .option("password", self.config.db_password) \
                    .option("driver", self.config.db_driver) \
                    .load()

                count = df.count()
                self.logger.info(f"Успешно выгружено {count} строк из витрины.")
                return df
            except Py4JJavaError as e:
                if attempt == retries:
                    self.logger.critical("Не удалось подключиться к СУБД после всех попыток.")
                    raise e
                error_msg = str(e.java_exception).split('\n')[0]
                self.logger.warning(f"Ожидание готовности данных/БД (Попытка {attempt}). Ошибка: {error_msg}")
                time.sleep(delay)

    def load_results(self, df: DataFrame) -> None:
        """Сохраняет результаты работы модели (Load)."""
        self.logger.info(f"Выгрузка результатов кластеризации в таблицу {self.config.target_table}...")

        columns_to_drop = ["raw_features", "features"]
        df_to_write = df.drop(*columns_to_drop)

        try:
            df_to_write.write \
                .format("jdbc") \
                .option("url", self.config.db_url) \
                .option("dbtable", self.config.target_table) \
                .option("user", self.config.db_user) \
                .option("password", self.config.db_password) \
                .option("driver", self.config.db_driver) \
                .mode("overwrite") \
                .save()
            self.logger.info("Результаты работы модели сохранены!")
        except Py4JJavaError as e:
            self.logger.error(f"Ошибка JDBC при записи результатов: {e.java_exception}")
            raise e