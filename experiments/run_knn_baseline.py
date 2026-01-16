import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
import pandas as pd

from src.data_load import load_ratings
from src.split import leave_one_out_by_time
from src.knn_item import ItemKNNRecommender
from src.metrics import precision_at_k, recall_at_k, ndcg_at_k


def evaluate(model, train: pd.DataFrame, test: pd.DataFrame, k: int = 10, max_users: int = 200):
    # limit users for speed while we’re building
    test_sample = test.head(max_users)

    p_list, r_list, n_list = [], [], []

    t0 = time.time()
    for _, row in test_sample.iterrows():
        user = int(row["userId"])
        true_item = int(row["movieId"])

        recs = model.recommend(user, k=k)

        p_list.append(precision_at_k(recs, true_item, k))
        r_list.append(recall_at_k(recs, true_item, k))
        n_list.append(ndcg_at_k(recs, true_item, k))

    infer_time = time.time() - t0

    return {
        "users_evaluated": len(test_sample),
        "precision@k": sum(p_list) / len(p_list),
        "recall@k": sum(r_list) / len(r_list),
        "ndcg@k": sum(n_list) / len(n_list),
        "inference_seconds_total": infer_time,
        "inference_ms_per_user": (infer_time / len(test_sample)) * 1000.0,
    }


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = leave_one_out_by_time(ratings)

    model = ItemKNNRecommender(top_n_similar_items=50)

    t0 = time.time()
    model.fit(train)
    train_time = time.time() - t0

    results = evaluate(model, train, test, k=10, max_users=200)

    print("ItemKNN baseline results")
    print("Train time (s):", round(train_time, 3))
    for k, v in results.items():
        print(f"{k}: {v}")
