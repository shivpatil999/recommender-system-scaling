import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_load import load_ratings
from src.split import leave_one_out_by_time
from src.mf_sgd import MFSGDRecommender

ratings = load_ratings()
train, _ = leave_one_out_by_time(ratings)

model = MFSGDRecommender(factors=32, epochs=10)
model.fit(train)

user_id = train["userId"].iloc[0]
recs = model.recommend(user_id, k=10)

print(f"\nTop-10 movie recommendations for user {user_id}:")
for i, m in enumerate(recs, 1):
    print(f"{i}. Movie ID {m}")
