import os
from abc import ABC

import pandas as pd
from pandas import DataFrame

from src.database import Database


class DatabaseCSV(Database):


    def __init__(self, data_path):
        self.data_path = data_path
        self.bronze_path = data_path + 'bronze.csv'
        self.silver_path = data_path + 'silver.csv'
        self.gold_path = data_path + 'silver.csv'

    def post_sale(self, sale: dict):
        data = sale
        df_new = pd.DataFrame(data)

        if os.path.isfile(self.data_path) and os.path.getsize(self.data_path) > 0:
            df = pd.read_csv(self.data_path)
            df = pd.concat([df, df_new])

            df = df.drop_duplicates(subset=['year_week', 'vegetable'], keep='last')
        else:
            df = df_new

        df.to_csv(self.data_path, index=False)

    def post_sales(self, sales: list[dict]):
        for sale in sales:
            self.post_sale(sale)

    def get_raw_sales(self) -> DataFrame:
        if os.path.isfile(self.data_path) and os.path.getsize(self.data_path) > 0:
            df = pd.read_csv(self.data_path)
            return df
        return DataFrame()

