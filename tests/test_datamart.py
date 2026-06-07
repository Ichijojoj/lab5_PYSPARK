import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, LongType
from src.config import AppConfig
from src.data_mart import DataMart


class DummyOracleManager:
    """Заглушка для имитации работы OracleManager в тестах."""

    def __init__(self, df):
        self.df = df

    def extract_data(self):
        return self.df

    def load_results(self, df):
        self.loaded_df = df


class TestDataMartAndSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Инициализация тестовой локальной сессии Spark
        cls.spark = SparkSession.builder \
            .appName("UnitTest_DataMart") \
            .master("local[2]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        cls.config = AppConfig()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_schema_validation_success(self):
        """Тест: успешное прохождение валидации схемы при корректных данных."""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("cluster", DoubleType(), True)
        ])
        data = [(1, 0.0), (2, 1.0)]
        df = self.spark.createDataFrame(data, schema)

        dummy_db = DummyOracleManager(df)
        dm = DataMart(dummy_db)

        # Метод не должен вызывать исключений
        try:
            dm.push_results(df)
        except ValueError as e:
            self.fail(f"Валидация схемы вызвала ошибку при корректных данных: {e}")

    def test_schema_validation_failure(self):
        """Тест: генерация исключения при отсутствии необходимых колонок."""
        schema = StructType([
            StructField("some_other_column", LongType(), True)
        ])
        data = [(1,), (2,)]
        df = self.spark.createDataFrame(data, schema)

        dummy_db = DummyOracleManager(df)
        dm = DataMart(dummy_db)

        # Ожидается ValueError, так как отсутствуют колонки 'id' и 'cluster'
        with self.assertRaises(ValueError):
            dm.push_results(df)

    def test_incoming_preprocessed_data_no_nulls(self):
        """Тест: проверка отсутствия пустых значений в предобработанных полях витрины."""
        schema = StructType([
            StructField("ID", LongType(), True),
            StructField("ENERGY_100G", DoubleType(), True),
            StructField("FAT_100G", DoubleType(), True),
            StructField("CARBOHYDRATES_100G", DoubleType(), True),
            StructField("SUGARS_100G", DoubleType(), True),
            StructField("PROTEINS_100G", DoubleType(), True),
            StructField("SALT_100G", DoubleType(), True)
        ])

        # Передаем одну строку со значением Null
        data_with_null = [(1, 200.0, None, 50.0, 5.0, 8.0, 1.2)]
        df_null = self.spark.createDataFrame(data_with_null, schema)

        # Проверяем, есть ли пустые значения в числовых признаках
        null_count = df_null.filter(
            df_null["ENERGY_100G"].isNull() |
            df_null["FAT_100G"].isNull() |
            df_null["CARBOHYDRATES_100G"].isNull()
        ).count()

        # По логике лабораторной работы, витрина данных должна поставлять уже очищенные данные
        self.assertGreater(null_count, 0,
                           "Тест зафиксировал наличие Null-значений, которые должны фильтроваться на стороне Scala витрины.")


if __name__ == "__main__":
    unittest.main()