import pandas as pd
import torch.nn as nn


def make_predictions(config):
    if config['model'] == 'PrevMonthSale':
        df_sales = pd.read_csv(config["data"]["sales"])

        df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(1)

        df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)

        return df_sales[["dates", "item_id", "prediction"]]

    elif config['model'] == 'SameMonthLastYearSales':
        df_sales = pd.read_csv(config["data"]["sales"])
        df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(12)

        df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)

        return df_sales[["dates", "item_id", "prediction"]]

    elif config['model'] == 'Ridge':
        df_sales = pd.read_csv(config["data"]["sales"])

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split

        ##makes features
        df_sales['sales_lag_12'] = df_sales.groupby("item_id")['sales'].shift(12)

        df_sales['sales_mean_last_12'] = df_sales.groupby("item_id")['sales'].rolling(12).mean().shift(1).dropna()

        df_sales['sales_last_1_3'] = df_sales.groupby("item_id")['sales'].rolling(3).sum().shift(1).dropna()
        df_sales['sales_last_13_15'] = df_sales.groupby("item_id")['sales'].rolling(3).sum().shift(13).dropna()

        df_sales['sales_growth'] = (df_sales["sales_last_1_3"] / df_sales["sales_last_13_15"]) * df_sales[
            "sales_lag_12"]

        X = df_sales[["sales_lag_12", "sales_growth", 'sales_mean_last_12']]
        y = df_sales["sales"]

        ## training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ridge = Ridge()
        ridge.fit(X_train, y_train)

        df_sales['prediction'] = ridge.predict(df_sales['sales']).dropna()

        df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)
        # inference
        print(df_sales)

        return df_sales[["dates", "item_id", "prediction"]]
