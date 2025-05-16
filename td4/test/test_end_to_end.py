from td4.script import get_data, clusterize_pages, p_clusters, train_page_cluster_predictor, clusterize_users, \
    u_clusters, train_click_predictor, evaluate_model

import pickle

def load_test_models():
    with open("models/page_cluster_model.pkl", "rb") as f:
        page_cluster_model = pickle.load(f)

    # Load page vectorizer
    with open("models/page_vectorizer.pkl", "rb") as f:
        page_vectorizer = pickle.load(f)

    # Load page cluster predictor
    with open("models/page_cluster_predictor.pkl", "rb") as f:
        page_cluster_predictor = pickle.load(f)

    # Load user cluster model
    with open("models/user_cluster_model.pkl", "rb") as f:
        user_cluster_model = pickle.load(f)

    # Load click predictor
    with open("models/click_predictor.pkl", "rb") as f:
        click_predictor = pickle.load(f)

    return page_cluster_model, page_vectorizer, page_cluster_predictor, user_cluster_model, click_predictor


def test_end_to_end():
    print("\n== Building page clusters ==")
    clusterize_pages(p_clusters)

    print("\n== Training page cluster predictor ==")
    train_page_cluster_predictor()

    print("\n== Building user clusters ==")
    clusterize_users(u_clusters)

    print("\n== Training click predictor ==")
    train_click_predictor()

    print("\n== Evaluating model ==")
    evaluate_model()





