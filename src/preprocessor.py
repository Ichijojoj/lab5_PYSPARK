import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


class DataPreprocessor:
    """Отвечает за загрузку, очистку и векторизацию признаков."""

    def __init__(self, spark: SparkSession, config):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self) -> DataFrame:
        self.logger.info(f"Загрузка данных из {self.config.data_path}")
        df = self.spark.read.csv(
            self.config.data_path,
            header=True,
            inferSchema=True,
            sep='\t',
            multiLine=True,
            escape='"'
        )
        return df

    def clean_data(self, df: DataFrame) -> DataFrame:
        self.logger.info("Очистка данных от пропусков и аномалий...")
        df_selected = df.select(self.config.feature_columns)
        for c in self.config.feature_columns:
            df_selected = df_selected.withColumn(c, col(c).cast("float"))
        df_clean = df_selected.dropna(how="any", subset=self.config.feature_columns)

        for c in ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'sugars_100g']:
            df_clean = df_clean.filter((col(c) >= 0) & (col(c) <= 100))

        return df_clean

    def build_feature_pipeline(self) -> Pipeline:
        self.logger.info("Сборка пайплайна признаков (VectorAssembler + StandardScaler)")
        assembler = VectorAssembler(
            inputCols=self.config.feature_columns,
            outputCol="raw_features"
        )
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        return Pipeline(stages=[assembler, scaler])