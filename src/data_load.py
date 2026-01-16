import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_ratings():
    path = DATA_DIR / "ratings.csv"
    return pd.read_csv(path)

if __name__ == "__main__":
    ratings = load_ratings()

    from split import leave_one_out_by_time
    train, test = leave_one_out_by_time(ratings)

    print("All ratings:", ratings.shape)
    print("Train:", train.shape)
    print("Test:", test.shape)

    print("Users in train:", train["userId"].nunique())
    print("Users in test:", test["userId"].nunique())

    # sanity checks
    print("Any users missing from train?",
          (set(test["userId"]) - set(train["userId"])) != set())

