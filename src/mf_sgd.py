import numpy as np
import pandas as pd


class MFSGDRecommender:
    """
    Explicit-feedback Matrix Factorization trained with SGD.

    Prediction:
      r_hat(u,i) = mu + bu[u] + bi[i] + P[u]·Q[i]

    Why this is the "advanced switch":
    - Training scales with #interactions (ratings), not with full similarity matrices.
    - Inference is fast (vector dot-products).
    """

    def __init__(
        self,
        factors: int = 32,
        lr: float = 0.01,
        reg: float = 0.05,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

        self.user_ids = None
        self.item_ids = None
        self.user_to_idx = None
        self.item_to_idx = None

        self.mu = 0.0
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None

        self.user_rated_items = None

    def fit(self, train: pd.DataFrame):
        rng = np.random.default_rng(self.seed)

        # index maps
        self.user_ids = np.array(sorted(train["userId"].unique()))
        self.item_ids = np.array(sorted(train["movieId"].unique()))
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {m: i for i, m in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        self.mu = float(train["rating"].mean())

        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.P = (0.1 * rng.standard_normal((n_users, self.factors))).astype(np.float32)
        self.Q = (0.1 * rng.standard_normal((n_items, self.factors))).astype(np.float32)

        # fast arrays
        u = train["userId"].map(self.user_to_idx).to_numpy()
        i = train["movieId"].map(self.item_to_idx).to_numpy()
        r = train["rating"].to_numpy(dtype=np.float32)

        # for filtering already-rated items in recommend()
        self.user_rated_items = train.groupby("userId")["movieId"].apply(set).to_dict()

        idx = np.arange(len(r))

        for _ in range(self.epochs):
            rng.shuffle(idx)
            for t in idx:
                uu = u[t]
                ii = i[t]
                rr = r[t]

                pred = self.mu + self.bu[uu] + self.bi[ii] + float(np.dot(self.P[uu], self.Q[ii]))
                err = rr - pred

                # biases update
                self.bu[uu] += self.lr * (err - self.reg * self.bu[uu])
                self.bi[ii] += self.lr * (err - self.reg * self.bi[ii])

                # latent updates
                pu = self.P[uu].copy()
                qi = self.Q[ii].copy()

                self.P[uu] += self.lr * (err * qi - self.reg * pu)
                self.Q[ii] += self.lr * (err * pu - self.reg * qi)

        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if user_id not in self.user_to_idx:
            return []

        uu = self.user_to_idx[user_id]

        # score all items
        scores = self.mu + self.bu[uu] + self.bi + (self.Q @ self.P[uu])

        # filter already-rated items
        rated = self.user_rated_items.get(user_id, set())
        if rated:
            rated_idx = [self.item_to_idx[m] for m in rated if m in self.item_to_idx]
            if rated_idx:
                scores[rated_idx] = -np.inf

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return self.item_ids[top_idx].tolist()
