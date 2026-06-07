import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AppConfig:
    app_name: str = "OpenFoodFacts_Clustering"
    data_path: str = "data/data.csv"
    model_save_path: str = "models/kmeans_food_model"
    feature_columns: List[str] = field(default_factory=lambda: [
        'energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g'
    ])
    k_clusters: int = 5
    random_seed: int = 42
    max_iter: int = 20

    _current_dir: str = field(default=os.path.dirname(os.path.abspath(__file__)), init=False, repr=False)
    spark_config_path: str = field(default="", init=False)

    def __post_init__(self):
        # Нахождение пути к файлу конфигурации относительно текущего скрипта
        base_dir = os.path.dirname(self._current_dir)
        self.spark_config_path = os.path.join(base_dir, "spark_config.json")

    def load_spark_config(self) -> Dict[str, str]:
        """Загружает параметры конфигурации Spark из внешнего JSON-файла."""
        logger = logging.getLogger(self.__class__.__name__)
        if not os.path.exists(self.spark_config_path):
            logger.warning(
                f"Файл конфигурации Spark не найден по пути: {self.spark_config_path}. "
                "Будут использованы параметры по умолчанию."
            )
            return {
                "spark.app.name": self.app_name,
                "spark.master": "local[*]",
                "spark.driver.memory": "2g"
            }

        try:
            with open(self.spark_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при разборе JSON-файла конфигурации Spark: {e}")
            raise