import logging
from lab5_PYSPARK.src.config import AppConfig
from lab5_PYSPARK.src.spark_manager import SparkManager
from lab5_PYSPARK.src.sanity_check import SanityChecker
from lab5_PYSPARK.src.preprocessor import DataPreprocessor
from lab5_PYSPARK.src.clustering import ClusteringModeler


class MLPipeline:
    """Оркестратор всего процесса машинного обучения."""

    def __init__(self):
        self.config = AppConfig()
        self.spark_manager = SparkManager(self.config.app_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        spark = self.spark_manager.spark
        try:
            # 1. Sanity Check (WordCount)
            checker = SanityChecker(spark)
            checker.run_wordcount()

            # 2. Подготовка данных
            preprocessor = DataPreprocessor(spark, self.config)
            raw_df = preprocessor.load_data()
            clean_df = preprocessor.clean_data(raw_df)

            # 3. Кешируем данные перед ML пайплайном для ускорения
            clean_df.cache()
            self.logger.info(f"Размер очищенной выборки: {clean_df.count()} строк.")

            # 4. Векторизация и масштабирование
            feature_pipeline = preprocessor.build_feature_pipeline()
            feature_model = feature_pipeline.fit(clean_df)
            ml_df = feature_model.transform(clean_df)

            # 5. Кластеризация
            modeler = ClusteringModeler(self.config)
            modeler.train(ml_df)
            modeler.evaluate(ml_df)
            modeler.save_model()

            self.logger.info("Пайплайн успешно завершен!")

        except Exception as e:
            self.logger.error(f"Критическая ошибка в пайплайне: {e}", exc_info=True)
            raise
        finally:
            # Освобождение ресурсов гарантируется блоком finally
            self.spark_manager.stop()