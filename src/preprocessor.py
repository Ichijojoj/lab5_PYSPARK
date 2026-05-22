import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


class DataPreprocessor:
    def __init__(self, spark: SparkSession, config):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_feature_pipeline(self) -> Pipeline:
        self.logger.info("Сборка пайплайна признаков")
        feature_cols_upper = [col.upper() for col in self.config.feature_columns]

        assembler = VectorAssembler(
            inputCols=feature_cols_upper,
            outputCol="raw_features"
        )
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        return Pipeline(stages=[assembler, scaler])