# Netflix-Style Recommendation System (Scaling Study)

This project implements a Netflix-style movie recommender system and demonstrates how recommendation algorithms behave as data scales.

## Project Overview
I first implemented a simple item-based kNN collaborative filtering model as a baseline. While it performs reasonably well on small datasets, inference latency increases rapidly as the number of users and items grows.

To address this, I switched to a Matrix Factorization model trained with Stochastic Gradient Descent (SGD). This shifts computational cost to offline training while enabling extremely fast online inference.

## Models Implemented
- Item-based kNN (cosine similarity)
- Matrix Factorization with SGD

## Evaluation Method
- Leave-one-out per-user split
- Precision@10, Recall@10, NDCG@10
- Training time, inference latency, and memory usage tracked

## Key Results
- kNN inference time grows significantly with dataset size
- Matrix Factorization reduces inference latency by over 100×
- Demonstrates why MF-based models are preferred in production recommender systems

## Results
![Inference Latency](results/plots/inference_latency.png)
![Training Time](results/plots/training_time.png)
![Recall@10](results/plots/recall.png)

## How to Run
```bash
pip install -r requirements.txt
python experiments/run_knn_scaling.py
python experiments/run_mf_scaling.py
python experiments/plot_results.py
