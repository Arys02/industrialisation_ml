import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from td4.pipeline.clustering import clusterize_pages
from td4.pipeline.features import build_click_features
from td4.pipeline.preprocessing import preprocess_text

from sklearn.linear_model import LogisticRegression

from td4.config import MODEL_DIR
from td4.utils.ClusterParameters import ClusterParameters


def train_page_cluster_predictor(_cache, cluster_parameters: ClusterParameters):
    page_data, _, vect = clusterize_pages(_cache, cluster_parameters.p_clusters)

    X_pages = vect.transform(preprocess_text(page_data['page_text']))
    y = page_data['cluster']

    lr = LogisticRegression(max_iter=1000, random_state=cluster_parameters.seed)
    lr.fit(X_pages, y)

    _cache["page_cluster_predictor"] = lr

    return lr

def train_click_predictor(_cache, cluster_parameters: ClusterParameters):
    click_features = build_click_features(_cache, cluster_parameters)

    X = click_features.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)

    y = click_features['clicked']

    lr = LogisticRegression(max_iter=1000, random_state=cluster_parameters.seed)
    lr.fit(X, y)

    _cache["click_predictor"] = lr

    return lr


def save_models(_cache):
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save page cluster model
    with open(MODEL_DIR / "page_cluster_model.pkl", "wb") as f:
        pickle.dump(_cache["page_cluster_model"], f)

    # Save page vectorizer
    with open(MODEL_DIR / "page_vectorizer.pkl", "wb") as f:
        pickle.dump(_cache["page_vectorizer"], f)

    # Save page cluster predictor
    with open(MODEL_DIR / "page_cluster_predictor.pkl", "wb") as f:
        pickle.dump(_cache["page_cluster_predictor"], f)

    # Save user cluster model
    with open(MODEL_DIR / "user_cluster_model.pkl", "wb") as f:
        pickle.dump(_cache["user_cluster_model"], f)

    # Save click predictor
    with open(MODEL_DIR / "click_predictor.pkl", "wb") as f:
        pickle.dump(_cache["click_predictor"], f)


def load_models(_cache):
    with open(MODEL_DIR / "page_cluster_model.pkl", "rb") as f:
        _cache["page_cluster_model"] = pickle.load(f)

    # Load page vectorizer
    with open(MODEL_DIR / "page_vectorizer.pkl", "rb") as f:
        _cache["page_vectorizer"] = pickle.load(f)

    # Load page cluster predictor
    with open(MODEL_DIR / "page_cluster_predictor.pkl", "rb") as f:
        _cache["page_cluster_predictor"] = pickle.load(f)

    # Load user cluster model
    with open(MODEL_DIR / "user_cluster_model.pkl", "rb") as f:
        _cache["user_cluster_model"] = pickle.load(f)

    # Load click predictor
    with open(MODEL_DIR / "click_predictor.pkl", "rb") as f:
        _cache["click_predictor"] = pickle.load(f)

    return _cache["page_cluster_model"], _cache["page_vectorizer"], _cache["page_cluster_predictor"], _cache[
        "user_cluster_model"], _cache["click_predictor"]

def evaluate_model(_cache, cluster_parameters: ClusterParameters):
    click_features = build_click_features(_cache, cluster_parameters)

    msk = np.random.rand(len(click_features)) < 0.8
    train = click_features[msk]
    test = click_features[~msk]

    X_train = train.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
    y_train = train['clicked']

    lr = LogisticRegression(max_iter=1000, random_state=cluster_parameters.seed)
    lr.fit(X_train, y_train)

    X_test = test.drop(['user_id', 'page_id', 'ad_id', 'clicked'], axis=1)
    y_test = test['clicked']

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")


