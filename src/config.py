from dataclasses import dataclass, field
from typing import List

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