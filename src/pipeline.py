import logging
from py4j.protocol import Py4JJavaError
from pyspark.errors import AnalysisException, IllegalArgumentException

from src.config import AppConfig
from src.spark_config import SparkConfig
from src.spark_manager import SparkManager
from src.preprocessor import DataPreprocessor
from src.clustering import ClusteringModeler
from src.oracle_manager import OracleManager

class MLPipeline:
    def __init__(self):
        self.config = AppConfig()
        self.spark_config = SparkConfig()
        self.spark_manager = SparkManager(self.spark_config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        spark = self.spark_manager.spark
        db_manager = OracleManager(spark, self.config)

        try:
            # Чтение готовой витрины
            cleaned_df = db_manager.extract_data()
            cleaned_df.cache()

            preprocessor = DataPreprocessor(spark, self.config)
            feature_pipeline = preprocessor.build_feature_pipeline()
            feature_model = feature_pipeline.fit(cleaned_df)
            ml_df = feature_model.transform(cleaned_df)

            modeler = ClusteringModeler(self.config)
            modeler.train(ml_df)
            modeler.evaluate(ml_df)
            modeler.save_model()

            predictions_df = modeler.model.transform(ml_df)
            db_manager.load_results(predictions_df)
            self.logger.info("Пайплайн модели на основе витрины успешно завершен!")

        except Py4JJavaError as db_err:
            self.logger.critical("Ошибка уровня JDBC/драйвера при работе с витриной данных.")
            self.logger.error(str(db_err.java_exception).split('\n')[0])
        except AnalysisException as sql_err:
            self.logger.critical("Несоответствие схемы таблицы-витрины.")
            self.logger.error(str(sql_err))
        except IllegalArgumentException as arg_err:
            self.logger.critical("Неверные аргументы конфигурации.")
            self.logger.error(str(arg_err))
        finally:
            self.spark_manager.stop()