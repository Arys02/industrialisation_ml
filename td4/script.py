import logging
import warnings

from td4.config import *
from td4.pipeline.clustering import clusterize_users, clusterize_pages
from td4.pipeline.dataset import get_data
from td4.pipeline.models import train_page_cluster_predictor, train_click_predictor, save_models, \
    evaluate_model

from td4.utils.cache import _cache

from td4.utils.ClusterParameters import ClusterParameters

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG, format=LOGS_FORMAT, filename=f'{LOGS_DIR}/app.log')


def main():
    cluster_parameter = ClusterParameters(u_clusters=5, p_clusters=7, seed=42)

    """Main function"""
    print("Starting ad prediction system...")
    get_data(_cache)

    print("\n== Building page clusters ==")
    clusterize_pages(_cache, cluster_parameter.p_clusters, seed=cluster_parameter.seed)

    print("\n== Training page cluster predictor ==")
    train_page_cluster_predictor(_cache, cluster_parameter)

    print("\n== Building user clusters ==")
    clusterize_users(_cache, cluster_parameter.u_clusters, seed=cluster_parameter.seed)

    print("\n== Training click predictor ==")
    train_click_predictor(_cache, cluster_parameter)

    print("\n== Evaluating model ==")
    evaluate_model(_cache, cluster_parameter)

    print("\n== Saving models ==")
    save_models(_cache)

    print("\nDone!")


if __name__ == "__main__":
    main()
