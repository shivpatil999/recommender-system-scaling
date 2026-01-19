import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
import pandas as pd

from src.data_load import load_ratings
from src.split import leave_one_out_by_time
from src.mf_sgd import MFSGDRecommender
from src.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.utils import get_process_memory_mb


def evaluate(model, test: pd.DataFrame, k: int = 10, max_users: int = 300):
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
        "precision@10": sum(p_list) / len(p_list),
        "recall@10": sum(r_list) / len(r_list),
        "ndcg@10": sum(n_list) / len(n_list),
        "inference_ms_per_user": (infer_time / len(test_sample)) * 1000.0,
    }


if __name__ == "__main__":
    ratings = load_ratings()

    sizes = [10_000, 25_000, 50_000, 75_000, 100_000]
    rows = []

    for n in sizes:
        subset = ratings.head(n).copy()
        train, test = leave_one_out_by_time(subset)

        mem_before = get_process_memory_mb()

        model = MFSGDRecommender(factors=32, lr=0.01, reg=0.05, epochs=10)

        t0 = time.time()
        model.fit(train)
        train_time = time.time() - t0

        mem_after_fit = get_process_memory_mb()

        metrics = evaluate(model, test, k=10, max_users=300)

        mem_after_eval = get_process_memory_mb()

        row = {
            "interactions": n,
            "users": train["userId"].nunique(),
            "items": train["movieId"].nunique(),
            "train_time_s": train_time,
            "mem_before_mb": mem_before,
            "mem_after_fit_mb": mem_after_fit,
            "mem_after_eval_mb": mem_after_eval,
            **metrics,
        }

        rows.append(row)

        print("\n--- MF scaling point ---")
        for k, v in row.items():
            print(f"{k}: {v}")

    out = pd.DataFrame(rows)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/mf_scaling.csv")
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
