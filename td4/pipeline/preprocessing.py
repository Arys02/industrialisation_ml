import pandas as pd

from td4.pipeline.dataset import get_data


def preprocess_text(text_series):
    text_series = text_series.fillna("")
    text_series = text_series.str.lower()
    return text_series


def process_user_data(_cache):
    """Process user data for clustering"""
    # Get data
    user_data, _, bid_data, _ = get_data(_cache)

    # One-hot encode user features
    user_processed = pd.get_dummies(user_data, columns=['sex', 'city', 'device'])

    # Join with bid data to get user-page interactions
    user_visits = (
        bid_data.groupby(["user_id", "page_id"])
        .size()
        .unstack(1)
        .fillna(0)
    )
    user_visits.columns = [str(c) for c in user_visits.columns]
    user_processed = user_processed.merge(user_visits, on='user_id', how='left')

    # Cache processed data
    _cache["processed_user_data"] = user_processed

    return user_processed
