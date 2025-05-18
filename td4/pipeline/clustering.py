from functools import cache

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from td4.pipeline.dataset import get_data
from td4.pipeline.preprocessing import process_user_data, preprocess_text


def clusterize_users(_cache, k=5, seed=42):
    if "user_clusters" in _cache:
        return _cache["user_clusters"], _cache["user_cluster_model"]

    user_processed = process_user_data(_cache)

    km = KMeans(n_clusters=k, random_state=seed)
    user_clusters = km.fit_predict(user_processed.drop('user_id', axis=1))

    user_processed['cluster'] = user_clusters

    _cache["user_clusters"] = user_processed
    _cache["user_cluster_model"] = km

    return user_processed, km

def clusterize_pages(_cache, k=7, seed=42):
    if "page_clusters" in _cache:
        return _cache["page_clusters"], _cache["page_cluster_model"], _cache["page_vectorizer"]

    _, page_data, _, _ = get_data(_cache)

    vect = TfidfVectorizer(max_features=1000, stop_words='english')
    X_pages = vect.fit_transform(preprocess_text(page_data['page_text']))

    km = KMeans(n_clusters=k, random_state=seed)
    page_clusters = km.fit_predict(X_pages)

    page_data['cluster'] = page_clusters

    _cache["page_clusters"] = page_data
    _cache["page_cluster_model"] = km
    _cache["page_vectorizer"] = vect

    return page_data, km, vect
