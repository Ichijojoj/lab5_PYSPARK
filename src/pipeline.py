import logging
from pyspark.errors import PySparkException, AnalysisException, IllegalArgumentException
from lab5_PYSPARK.src.config import AppConfig
from lab5_PYSPARK.src.spark_manager import SparkManager
from lab5_PYSPARK.src.sanity_check import SanityChecker
from lab5_PYSPARK.src.preprocessor import DataPreprocessor
from lab5_PYSPARK.src.clustering import ClusteringModeler


class MLPipeline:
    """Оркестратор всего процесса машинного обучения."""

    def __init__(self):
        self.config = AppConfig()
        # Считывание параметров Spark из конфигурационного файла
        spark_configs = self.config.load_spark_config()
        self.spark_manager = SparkManager(spark_configs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        spark = self.spark_manager.spark
        try:
            # Sanity Check (WordCount)
            checker = SanityChecker(spark)
            checker.run_wordcount()

            # данные
            preprocessor = DataPreprocessor(spark, self.config)
            raw_df = preprocessor.load_data()
            clean_df = preprocessor.clean_data(raw_df)

            # кеш перед пайплайном для ускорения
            clean_df.cache()
            self.logger.info(f"Размер очищенной выборки: {clean_df.count()} строк.")

            #векторизация и масштабирование
            feature_pipeline = preprocessor.build_feature_pipeline()
            feature_model = feature_pipeline.fit(clean_df)
            ml_df = feature_model.transform(clean_df)

            #кластеризация
            modeler = ClusteringModeler(self.config)
            modeler.train(ml_df)
            modeler.evaluate(ml_df)
            modeler.save_model()

            self.logger.info("Пайплайн успешно завершен!")

        except FileNotFoundError as e:
            self.logger.error(f"Не найден файл или путь к данным: {e}")
            raise
        except AnalysisException as e:
            self.logger.error(f"Ошибка анализа Spark SQL (проверьте схему, разделители или типы колонок): {e}")
            raise
        except IllegalArgumentException as e:
            self.logger.error(f"Некорректные аргументы при настройке алгоритма ML / сборщика признаков: {e}")
            raise
        except PySparkException as e:
            self.logger.error(f"Внутренняя ошибка выполнения среды PySpark: {e}")
            raise
        except KeyboardInterrupt:
            self.logger.warning("Процесс выполнения был прерван пользователем.")
            raise
        except Exception as e:
            self.logger.error(f"Непредвиденная системная ошибка в пайплайне: {e}", exc_info=True)
            raise
        finally:
            self.spark_manager.stop()