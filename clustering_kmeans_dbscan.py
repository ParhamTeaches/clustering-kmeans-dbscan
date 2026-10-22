
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


X, true_labels = make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

noise = np.random.uniform(
    low=-10,
    high=10,
    size=(40, 2)
)

X = np.vstack([X, noise])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(7, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    s=30
)

plt.title("Original Data")
plt.xlabel("Feature 1 scaled")
plt.ylabel("Feature 2 scaled")
plt.show()


k_values = [2, 3, 4, 5]

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print("\n" + "=" * 50)
    print(f"K-Means with k = {k}")
    print("=" * 50)
    print("Silhouette Score:", score)

    plt.figure(figsize=(7, 5))

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        s=30
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroids"
    )

    plt.title(f"K-Means Clustering with k = {k}")
    plt.xlabel("Feature 1 scaled")
    plt.ylabel("Feature 2 scaled")
    plt.legend()
    plt.show()


dbscan_parameters = [
    {"eps": 0.2, "min_samples": 5},
    {"eps": 0.3, "min_samples": 5},
    {"eps": 0.4, "min_samples": 5},
    {"eps": 0.4, "min_samples": 10}
]

for params in dbscan_parameters:
    eps = params["eps"]
    min_samples = params["min_samples"]

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_scaled)

    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    number_of_noise_points = list(labels).count(-1)

    print("\n" + "=" * 50)
    print(f"DBSCAN with eps = {eps}, min_samples = {min_samples}")
    print("=" * 50)
    print("Number of clusters:", number_of_clusters)
    print("Number of noise points:", number_of_noise_points)

    mask = labels != -1
    clustered_labels = labels[mask]

    if (
        len(set(clustered_labels)) > 1
        and len(clustered_labels) > len(set(clustered_labels))
    ):
        score = silhouette_score(X_scaled[mask], clustered_labels)
        print("Silhouette Score (excluding noise):", score)
    else:
        print("Silhouette Score: Not available (fewer than two valid clusters)")

    plt.figure(figsize=(7, 5))

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        s=30
    )

    plt.title(f"DBSCAN: eps={eps}, min_samples={min_samples}")
    plt.xlabel("Feature 1 scaled")
    plt.ylabel("Feature 2 scaled")
    plt.show()
