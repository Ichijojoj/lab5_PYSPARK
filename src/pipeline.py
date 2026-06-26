import logging
import os
import resource
import psutil
from py4j.protocol import Py4JJavaError
from pyspark.errors import AnalysisException, IllegalArgumentException

from src.config import AppConfig
from src.spark_config import SparkConfig
from src.spark_manager import SparkManager
from src.preprocessor import DataPreprocessor
from src.clustering import ClusteringModeler
from src.oracle_manager import OracleManager
from src.data_mart import DataMart


class MLPipeline:
    """Основной оркестратор ML-процесса в Kubernetes с профилированием ресурсов."""

    def __init__(self):
        self.config = AppConfig()
        self.spark_config = SparkConfig()
        self.spark_manager = SparkManager(self.spark_config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _log_resource_utilization(self) -> None:
        """Сбор и логирование показателей утилизации ресурсов процессом."""
        try:
            process = psutil.Process(os.getpid())

            # Текущее потребление памяти процессом
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)

            # Пиковое потребление памяти (Max RSS) всей группы процессов на уровне ОС
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # На Linux resource.ru_maxrss возвращается в килобайтах
            peak_rss_mb = usage.ru_maxrss / 1024.0

            # Статистика по процессору
            cpu_times = process.cpu_times()
            user_cpu_time = cpu_times.user
            system_cpu_time = cpu_times.system

            self.logger.info("=" * 50)
            self.logger.info("📊 ОТЧЕТ ОБ УТИЛИЗАЦИИ РЕСУРСОВ КОНТЕЙНЕРА (Job):")
            self.logger.info(f"   - Текущая физическая память (RSS): {rss_mb:.2f} MB")
            self.logger.info(f"   - Виртуальная память (VMS): {vms_mb:.2f} MB")
            self.logger.info(f"   - Пиковая физическая память (Max RSS): {peak_rss_mb:.2f} MB")
            self.logger.info(f"   - Время CPU (пользовательское): {user_cpu_time:.2f} сек")
            self.logger.info(f"   - Время CPU (системное): {system_cpu_time:.2f} сек")
            self.logger.info("=" * 50)
        except Exception as e:
            self.logger.warning(f"Не удалось собрать метрики утилизации ресурсов: {e}")

    def run(self) -> None:
        spark = self.spark_manager.spark
        db_manager = OracleManager(spark, self.config)
        data_mart = DataMart(db_manager)

        try:
            # Чтение подготовленной витрины с валидацией схемы
            cleaned_df = data_mart.get_prepared_data()
            cleaned_df.cache()

            # Предобработка признаков
            preprocessor = DataPreprocessor(spark, self.config)
            feature_pipeline = preprocessor.build_feature_pipeline()
            feature_model = feature_pipeline.fit(cleaned_df)
            ml_df = feature_model.transform(cleaned_df)

            # Обучение и оценка модели кластеризации
            modeler = ClusteringModeler(self.config)
            modeler.train(ml_df)
            modeler.evaluate(ml_df)
            modeler.save_model()

            # Предсказание и сохранение результатов
            predictions_df = modeler.model.transform(ml_df)
            data_mart.push_results(predictions_df)

            self.logger.info("Пайплайн модели на основе витрины в K8s успешно завершен!")

        except Py4JJavaError as db_err:
            self.logger.critical("Сетевой сбой СУБД или ошибка уровня JDBC/драйвера при работе с K8s-сервисом БД.")
            self.logger.error(f"Детали ошибки Java: {str(db_err.java_exception).splitlines()[0]}")
            raise
        except AnalysisException as sql_err:
            self.logger.critical("Несоответствие структуры таблицы-витрины (схема БД не совпадает с конфигурацией).")
            self.logger.error(str(sql_err))
            raise
        except IllegalArgumentException as arg_err:
            self.logger.critical("Неверные аргументы конфигурации алгоритмов или Spark-контекста.")
            self.logger.error(str(arg_err))
            raise
        except ValueError as val_err:
            self.logger.error(f"Ошибка валидации данных на уровне бизнес-логики: {val_err}")
            raise
        finally:
            # Сбор метрик перед выгрузкой контекста
            self._log_resource_utilization()
            self.spark_manager.stop()