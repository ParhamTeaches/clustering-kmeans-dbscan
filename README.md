# K-Means and DBSCAN Clustering Comparison

A Python machine learning project developed as part of AI coursework. It explores two unsupervised learning algorithms, K-Means and DBSCAN, using synthetic two-dimensional data.

## Dataset

- 500 synthetic samples generated using `make_blobs` with three centers.
- 40 additional randomly generated points.
- Feature standardization using `StandardScaler`.

## Algorithms and Experiments

**K-Means**

- Tests four cluster counts: 2, 3, 4, and 5.
- Calculates the Silhouette Score for each configuration.
- Visualizes the resulting clusters and their centroids.

**DBSCAN**

- Tests four combinations of `eps` and `min_samples`.
- Reports the number of detected clusters and noise points.
- Calculates the Silhouette Score using non-noise points when at least two valid clusters are available.
- Visualizes the clustering results.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Python script:

```bash
python clustering_kmeans_dbscan.py
```

The script displays clustering visualizations and prints evaluation results to the console.

## Evaluation Notes

Silhouette Scores for K-Means are calculated using all data points, while DBSCAN scores exclude points labeled as noise. These scores therefore evaluate different subsets of the data and should not be treated as directly equivalent.

## Project Purpose

This educational project demonstrates basic clustering techniques, parameter experimentation, and visualization using scikit-learn and Matplotlib.