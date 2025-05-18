import pandas as pd

from td4.pipeline.clustering import clusterize_users, clusterize_pages
from td4.pipeline.dataset import get_data
from td4.pipeline.preprocessing import preprocess_text
from td4.utils.ClusterParameters import ClusterParameters


def build_click_features(_cache, cluster_parameters: ClusterParameters):
    """Build features for click prediction"""
    user_data, page_data, bid_data, click_data = get_data(_cache)

    # Number of ad seen this day before this page
    click_data["date"] = click_data["timestamp"].apply(lambda txt: txt[:10])
    click_data["count"] = 1
    click_data["user_ads_seen"] = (
        click_data.groupby(["user_id", "date"])["count"]
        .cumsum()
    )

    click_data = click_data[["user_id", "page_id", "ad_id", "user_ads_seen", "clicked"]]

    user_clusters, _ = clusterize_users(_cache, cluster_parameters.u_clusters, seed=cluster_parameters.seed)
    page_clusters, _, _ = clusterize_pages(_cache, cluster_parameters.p_clusters, seed=cluster_parameters.seed)

    click_features = click_data.merge(user_clusters[['user_id', 'cluster']], on='user_id', how='left')
    click_features = click_features.rename(columns={'cluster': 'user_cluster'})

    cluster_probs = []
    page_to_cluster_prob = {page_id: get_page_cluster_probabilities(_cache, page_id, cluster_parameters) for page_id in
                            click_features["page_id"].unique()}

    cluster_probs = [page_to_cluster_prob[page_id] for page_id in click_features["page_id"]]

    cluster_prob_df = pd.DataFrame(
        cluster_probs,
        columns=[f'page_cluster_prob_{i}' for i in range(cluster_parameters.p_clusters)],
    )

    click_features = pd.concat(
        [click_features.reset_index(drop=True), cluster_prob_df.reset_index(drop=True)],
        axis=1,
    )

    _cache["click_features"] = click_features

    return click_features

def get_page_cluster_probabilities(_cache, page_id, cluster_parameters: ClusterParameters):
    """Get probabilities of a page belonging to each cluster"""
    page_data, _, vect = clusterize_pages(_cache, cluster_parameters.p_clusters)

    lr = _cache.get("page_cluster_predictor")
    if not lr:
        from td4.pipeline.models import train_page_cluster_predictor
        lr = train_page_cluster_predictor(_cache, cluster_parameters)

    page_text = page_data[page_data['page_id'] == page_id]['page_text'].values[0]

    X = vect.transform([preprocess_text(pd.Series([page_text]))[0]])

    probs = lr.predict_proba(X)[0]

    return probs
