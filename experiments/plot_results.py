import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# load results
knn = pd.read_csv("results/logs/knn_scaling.csv")
mf = pd.read_csv("results/logs/mf_scaling.csv")

Path("results/plots").mkdir(parents=True, exist_ok=True)

# -------- Plot 1: Inference latency --------
plt.figure()
plt.plot(knn["interactions"], knn["inference_ms_per_user"], marker="o", label="Item-kNN")
plt.plot(mf["interactions"], mf["inference_ms_per_user"], marker="o", label="Matrix Factorization (SGD)")
plt.xlabel("Number of interactions")
plt.ylabel("Inference time (ms/user)")
plt.title("Inference Latency Scaling")
plt.legend()
plt.savefig("results/plots/inference_latency.png")
plt.close()

# -------- Plot 2: Training time --------
plt.figure()
plt.plot(knn["interactions"], knn["train_time_s"], marker="o", label="Item-kNN")
plt.plot(mf["interactions"], mf["train_time_s"], marker="o", label="Matrix Factorization (SGD)")
plt.xlabel("Number of interactions")
plt.ylabel("Training time (seconds)")
plt.title("Training Time Scaling")
plt.legend()
plt.savefig("results/plots/training_time.png")
plt.close()

# -------- Plot 3: Recall@10 --------
plt.figure()
plt.plot(knn["interactions"], knn["recall@10"], marker="o", label="Item-kNN")
plt.plot(mf["interactions"], mf["recall@10"], marker="o", label="Matrix Factorization (SGD)")
plt.xlabel("Number of interactions")
plt.ylabel("Recall@10")
plt.title("Recommendation Quality (Recall@10)")
plt.legend()
plt.savefig("results/plots/recall.png")
plt.close()

print("Plots saved in results/plots/")
