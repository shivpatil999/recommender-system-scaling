import pandas as pd

def leave_one_out_by_time(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, hold out their most recent interaction as test (by timestamp).
    Everything else goes to train.

    Assumes columns: userId, movieId, rating, timestamp
    """
    ratings = ratings.sort_values(["userId", "timestamp"])
    # last row per user -> test
    test_idx = ratings.groupby("userId").tail(1).index
    test = ratings.loc[test_idx].copy()
    train = ratings.drop(index=test_idx).copy()

    return train, test
