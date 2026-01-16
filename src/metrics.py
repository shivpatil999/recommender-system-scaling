import numpy as np

def precision_at_k(recommended_items, relevant_item, k: int) -> float:
    """
    recommended_items: list of movieIds ranked by relevance
    relevant_item: the true test movieId
    """
    if relevant_item in recommended_items[:k]:
        return 1.0 / k
    return 0.0


def recall_at_k(recommended_items, relevant_item, k: int) -> float:
    if relevant_item in recommended_items[:k]:
        return 1.0
    return 0.0


def ndcg_at_k(recommended_items, relevant_item, k: int) -> float:
    """
    Since there is only ONE relevant item, NDCG simplifies a lot.
    """
    if relevant_item in recommended_items[:k]:
        rank = recommended_items.index(relevant_item) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0
if __name__ == "__main__":
    recs = [10, 20, 30, 40, 50]
    true_item = 30

    print("Precision@5:", precision_at_k(recs, true_item, 5))
    print("Recall@5:", recall_at_k(recs, true_item, 5))
    print("NDCG@5:", ndcg_at_k(recs, true_item, 5))
