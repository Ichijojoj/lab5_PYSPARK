import logging
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator


class ClusteringModeler:
    """Управление алгоритмом кластеризации (K-Means)."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None

    def train(self, df: DataFrame) -> KMeansModel:
        self.logger.info(f"Обучение K-Means (k={self.config.k_clusters})...")
        kmeans = KMeans() \
            .setK(self.config.k_clusters) \
            .setSeed(self.config.random_seed) \
            .setMaxIter(self.config.max_iter) \
            .setFeaturesCol("features") \
            .setPredictionCol("cluster")

        self.model = kmeans.fit(df)
        return self.model

    def evaluate(self, df: DataFrame) -> float:
        self.logger.info("Оценка модели кластеризации...")
        predictions = self.model.transform(df)
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        silhouette = evaluator.evaluate(predictions)
        self.logger.info(f"Silhouette (Силуэт) метрика: {silhouette:.4f}")
        return silhouette

    def save_model(self):
        if self.model:
            self.logger.info(f"Сохранение модели в {self.config.model_save_path}")
            self.model.write().overwrite().save(self.config.model_save_path)