import numpy as np

from td4.pipeline.clustering import clusterize_users
from td4.pipeline.features import get_page_cluster_probabilities
from td4.pipeline.models import train_click_predictor, load_models
from td4.utils.ClusterParameters import ClusterParameters


def predict_click(_cache, user_id, page_id, ad_id, cluster_parameters: ClusterParameters):
    user_clusters, _ = clusterize_users(cluster_parameters.u_clusters, seed=cluster_parameters.seed)
    user_cluster = user_clusters[user_clusters['user_id'] == user_id]['cluster'].values[0]

    page_probs = get_page_cluster_probabilities(_cache, page_id, cluster_parameters)

    features = np.hstack([np.array([user_cluster]), page_probs, np.array([ad_id])])

    lr = train_click_predictor(_cache, cluster_parameters)

    prob = lr.predict_proba(features.reshape(1, -1))[0][1]

    return prob



def get_recommendations(_cache, user_id, page_id, ad_ids, cluster_parameters: ClusterParameters):
    load_models(_cache)
    predictions = []
    for ad_id in ad_ids:
        prob = predict_click(user_id, page_id, ad_id, cluster_parameters)
        predictions.append((ad_id, prob))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions
