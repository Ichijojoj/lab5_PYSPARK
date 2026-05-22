import logging
from pyspark.sql import DataFrame

class DataMart:
    """Витрина данных: слой абстракции между БД и ML"""
    def __init__(self, oracle_manager):
        self.db = oracle_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_prepared_data(self) -> DataFrame:
        """ протокол получения данных из источника"""
        try:
            df = self.db.extract_data()
            if df is None:
                raise ValueError("Витрина вернула пустой набор данных.")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка при запросе к витрине: {e}")
            raise

    def push_results(self, df: DataFrame):
        """ проверка формата перед INSERT"""
        self._validate_schema(df)
        self.db.load_results(df)

    def _validate_schema(self, df: DataFrame):
        """Проверка структуры перед записью"""
        required_cols = {'id', 'cluster'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Ошибка формата данных! Ожидались колонки: {required_cols}")
        self.logger.info("Проверка схемы данных пройдена успешно.")