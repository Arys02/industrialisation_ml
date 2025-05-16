import logging

import pandas as pd

from td4.config import RAW_DATA_DIR


def get_data(cache=None):
    if cache is None:
        cache = {}

    logging.info(f"Getting data at {RAW_DATA_DIR} and put into cache")
    tmp_user_data = pd.read_csv(RAW_DATA_DIR / "user_data.csv")
    tmp_page_data = pd.read_csv(RAW_DATA_DIR / "page_data.csv")
    tmp_bid_data = pd.read_csv(RAW_DATA_DIR / "bid_requests_train.csv")
    tmp_click_data = pd.read_csv(RAW_DATA_DIR / "click_data_train.csv")

    cache["user_data"] = tmp_user_data
    cache["page_data"] = tmp_page_data
    cache["bid_data"] = tmp_bid_data
    cache["click_data"] = tmp_click_data

    return tmp_user_data, tmp_page_data, tmp_bid_data, tmp_click_data
