from pandas import DataFrame


def make_features(df_sales) -> (DataFrame, DataFrame):
    df_sales['sales_lag_12'] = df_sales.groupby("item_id")['sales'].shift(12)

    df_sales['sales_mean_last_12'] = df_sales.groupby("item_id")['sales'].rolling(12).mean()

    df_sales['sales_last_1_3'] = df_sales.groupby("item_id")['sales'].rolling(3).sum().shift(1)
    df_sales['sales_last_13_15'] = df_sales.groupby("item_id")['sales'].rolling(3).sum().shift(13)

    df_sales['sales_growth'] = (df_sales["sales_last_1_3"] / df_sales["sales_last_13_15"]) * df_sales[
        "sales_lag_12"]

    X = df_sales[["sales_lag_12", "sales_growth", 'sales_mean_last_12']]
    y = df_sales["sales"]
    return X, y
