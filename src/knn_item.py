import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNRecommender:
    """
    Item-based collaborative filtering using cosine similarity on the user-item rating matrix.
    """

    def __init__(self, top_n_similar_items: int = 50):
        self.top_n_similar_items = top_n_similar_items

        self.user_ids = None
        self.item_ids = None
        self.user_to_idx = None
        self.item_to_idx = None

        self.R = None  # user-item sparse matrix
        self.S = None  # item-item similarity matrix (dense for now)

    def fit(self, train_ratings: pd.DataFrame):
        # Build index mappings
        self.user_ids = np.array(sorted(train_ratings["userId"].unique()))
        self.item_ids = np.array(sorted(train_ratings["movieId"].unique()))

        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {m: i for i, m in enumerate(self.item_ids)}

        # Build sparse user-item matrix
        rows = train_ratings["userId"].map(self.user_to_idx).to_numpy()
        cols = train_ratings["movieId"].map(self.item_to_idx).to_numpy()
        vals = train_ratings["rating"].to_numpy(dtype=np.float32)

        self.R = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )

        # Item-item cosine similarity
        # cosine_similarity on sparse returns dense by default.
        self.S = cosine_similarity(self.R.T)

        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """
        Recommend top-k movieIds for a user.
        Strategy:
          score(item) = sum(sim(item, j) * rating(user, j)) over items j the user rated
        """
        if user_id not in self.user_to_idx:
            return []

        uidx = self.user_to_idx[user_id]
        user_row = self.R[uidx]

        # items the user has already rated
        rated_item_indices = user_row.indices
        rated_scores = user_row.data

        if len(rated_item_indices) == 0:
            return []

        # compute scores for all items
        # S[:, rated_items] -> similarity of every item to the items user rated
        sims = self.S[:, rated_item_indices]  # shape: (num_items, num_rated)
        scores = sims @ rated_scores  # shape: (num_items,)

        # don’t recommend already-rated items
        scores[rated_item_indices] = -np.inf

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return self.item_ids[top_idx].tolist()
