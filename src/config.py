import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    """Конфигурация приложения с поддержкой переменных окружения Kubernetes."""

    # Пути сохранения артефактов модели (поддерживает S3/HDFS/Local PV)
    model_save_path: str = os.getenv("MODEL_SAVE_PATH", "models/kmeans_food_model")

    # Фичи для кластеризации
    feature_columns: List[str] = field(default_factory=lambda: [
        'energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g'
    ])

    k_clusters: int = int(os.getenv("K_CLUSTERS", "5"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    max_iter: int = int(os.getenv("MAX_ITER", "20"))

    # Параметры подключения к БД (передаются через k8s Secrets и ConfigMaps)
    db_host: str = os.getenv("DB_HOST", "oracle-service")
    db_port: str = os.getenv("DB_PORT", "1521")
    db_sid: str = os.getenv("DB_SID", "XEPDB1")

    db_user: str = os.getenv("DB_USER", "SYSTEM")
    db_password: str = os.getenv("DB_PASSWORD", "oracle_password")
    db_driver: str = "oracle.jdbc.driver.OracleDriver"

    # Таблицы-источники и приемники
    source_table: str = os.getenv("SOURCE_TABLE", "SYSTEM.PREPROCESSED_FOOD_DATA")
    target_table: str = os.getenv("TARGET_TABLE", "SYSTEM.FOOD_CLUSTERS_RESULT")

    @property
    def db_url(self) -> str:
        """Формирует JDBC URL динамически на основе сетевых параметров k8s."""
        return f"jdbc:oracle:thin:@{self.db_host}:{self.db_port}/{self.db_sid}"