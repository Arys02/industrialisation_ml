from abc import ABC, abstractmethod

from pandas import DataFrame


class Database(ABC):
    @abstractmethod
    def post_sales(self, sale: list[dict]):
        pass

    @abstractmethod
    def get_raw_sales(self) -> DataFrame:
        pass

