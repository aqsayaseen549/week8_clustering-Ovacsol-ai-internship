
# ============================================================
# WEEK 8: UNSUPERVISED LEARNING & FEATURE ENGINEERING
# Project: Customer Segmentation using Clustering
# Dataset: Built-in Breast Cancer Dataset (features only)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Create output directory
os.makedirs("results", exist_ok=True)

# 2. Load dataset
dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

print("=" * 60)
print("WEEK 8: UNSUPERVISED LEARNING")
print("=" * 60)

print("\nDataset shape:", X.shape)
print("Missing values:", X.isnull().sum().sum())

# 3. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed.")

# 4. Find a suitable number of clusters using the elbow method
inertias = []
k_values = range(2, 9)

for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/elbow_method.png", dpi=300)
plt.show()

# 5. Train K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")
print("K-Means cluster counts:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

kmeans_silhouette = silhouette_score(
    X_scaled, kmeans_labels
)
print("K-Means silhouette score:",
      round(kmeans_silhouette, 4))

# 6. Hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

print("\nHierarchical clustering completed.")
print("Hierarchical cluster counts:")
print(
    pd.Series(hierarchical_labels)
    .value_counts()
    .sort_index()
)

hierarchical_silhouette = silhouette_score(
    X_scaled, hierarchical_labels
)
print("Hierarchical silhouette score:",
      round(hierarchical_silhouette, 4))

# 7. Create a dendrogram using a sample for readability
sample_size = min(100, len(X_scaled))
Z = linkage(
    X_scaled[:sample_size],
    method="ward"
)

plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.savefig("results/dendrogram.png", dpi=300)
plt.show()

# 8. PCA: reduce 30 features to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_.sum()

print("\nPCA completed.")
print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_pca.shape[1])
print(
    "Variance explained by 2 components:",
    round(explained_variance * 100, 2), "%"
)

# 9. Visualize K-Means clusters in PCA space
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    alpha=0.75
)

plt.title("K-Means Clusters Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("results/pca_clusters.png", dpi=300)
plt.show()

# 10. Compare clustering methods
comparison = pd.DataFrame([
    {
        "Method": "K-Means",
        "Clusters": len(set(kmeans_labels)),
        "Silhouette Score": kmeans_silhouette
    },
    {
        "Method": "Hierarchical",
        "Clusters": len(set(hierarchical_labels)),
        "Silhouette Score": hierarchical_silhouette
    }
])

print("\nCLUSTERING COMPARISON")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "results/clustering_comparison.csv",
    index=False
)

# 11. Save cluster assignments
cluster_results = X.copy()
cluster_results["KMeans_Cluster"] = kmeans_labels
cluster_results["Hierarchical_Cluster"] = hierarchical_labels

cluster_results.to_csv(
    "results/cluster_assignments.csv",
    index=False
)

# 12. Feature engineering example: create a combined feature
cluster_results["mean_radius_texture"] = (
    X["mean radius"] + X["mean texture"]
) / 2

cluster_results.to_csv(
    "results/cluster_assignments.csv",
    index=False
)

print("\nFeature engineering completed.")
print("Results saved in the results folder.")
print("\nWeek 8 project completed successfully!")
