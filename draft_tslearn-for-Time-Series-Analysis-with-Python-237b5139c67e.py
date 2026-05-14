# Description: Short example for tslearn for Time Series Analysis with Python.



from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from tslearn.clustering import KShape
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import (
    TimeSeriesScalerMeanVariance,
    TimeSeriesResampler
)
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Three sequences, five time steps, one feature
data = np.array([
    [[1], [2], [3], [4], [5]],
    [[2], [3], [4], [5], [6]],
    [[3], [4], [5], [6], [7]],
])

logger.info(data.shape)  # (3, 5, 1)


resampler = TimeSeriesResampler(sz=10)
resampled = resampler.fit_transform(data)
logger.info(resampled.shape)  # (3, 10, 1)


X = np.random.rand(200, 50, 1)
y = np.random.randint(0, 2, size=200)

# Chronological train/test split (first 80% train, last 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = TimeSeriesScalerMeanVariance()
scaler.fit(X_train)           # learn scaling on train portion only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


pipeline = make_pipeline(
    TimeSeriesScalerMeanVariance(),
    # downstream estimator (added later)
)


X = np.random.rand(100, 50, 1)
model = KShape(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

for centroid in model.cluster_centers_:
    plt.plot(centroid.ravel())
plt.title("K-Shape cluster centroids")
plt.show()


dtw_model = TimeSeriesKMeans(
    n_clusters=3,
    metric="dtw",
    random_state=42,
)
labels = dtw_model.fit_predict(X)


X = np.random.rand(200, 50, 1)
y = np.random.randint(0, 2, size=200)

clf = make_pipeline(
    TimeSeriesScalerMeanVariance(),
    KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw"),
)

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(clf, X, y, cv=tscv)
logger.info(f"Mean CV accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

# Optional: inspect the most recent fold as a hold-out
train_idx, test_idx = list(tscv.split(X))[-1]
clf.fit(X[train_idx], y[train_idx])
logger.info(f"Hold-out accuracy: {clf.score(X[test_idx], y[test_idx]):.2f}")


ts1 = np.array([[1], [2], [3], [4], [5]])
ts2 = np.array([[2], [3], [4], [5], [6]])

logger.info(f"DTW distance: {dtw(ts1, ts2):.2f}")


sax = SymbolicAggregateApproximation(
    n_segments=5,
    alphabet_size_avg=3,
)
X_sax = sax.fit_transform(X)
logger.info(X_sax[:2])


# Generate synthetic technology adoption data
np.random.seed(42)
countries = ['USA', 'UK', 'GER', 'FRA', 'JPN', 'CHN', 'IND', 'BRA']
years = np.arange(2000, 2020)
n_countries = len(countries)

# Create adoption patterns with different trajectories
data_rows = []
for country in countries:
    # Different countries have different adoption patterns
    base_rate = np.random.uniform(0.1, 0.5)
    trend = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, 0.02, len(years))
    
    # Create adoption curve
    adoption = base_rate + trend * np.arange(len(years)) + noise
    adoption = np.clip(adoption, 0, 1)  # Keep in [0, 1] range
    
    for year, value in zip(years, adoption):
        data_rows.append({
            'country': country,
            'year': year,
            'adoption_rate': value
        })

df = pd.DataFrame(data_rows)

# Pivot to time series format
subset = (
    df.pivot_table(index='country', columns='year', values='adoption_rate', aggfunc='mean')
      .sort_index(axis='columns')
)

# Forward-fill gaps, then replace any remaining missing values with zeros
subset = subset.ffill(axis=1).bfill(axis=1).fillna(0.0)

# Convert to the 3D array expected by tslearn: (samples, timesteps, features)
X = subset.to_numpy()[:, :, None]

# Optional: resample each series to the same number of points (e.g., 20 years)
resampler = TimeSeriesResampler(sz=20)
X_resampled = resampler.fit_transform(X)

# Scale per series to remove amplitude differences before clustering
scaler = TimeSeriesScalerMeanVariance()
X_scaled = scaler.fit_transform(X_resampled)

# Cluster countries by their adoption trajectory using DTW-aware k-means
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0, n_init=10)
labels = model.fit_predict(X_scaled)

clusters = pd.Series(labels, index=subset.index, name="cluster").sort_values()
logger.info("Cluster assignments:")
logger.info(clusters)

# Plot cluster centroids returned by tslearn
plt.figure(figsize=(10, 4))
for idx, center in enumerate(model.cluster_centers_):
    plt.plot(center.ravel(), label=f"Cluster {idx}", linewidth=2)
plt.title("Technology Adoption Cluster Centroids (DTW K-Means)")
plt.xlabel("Resampled Year Index")
plt.ylabel("Scaled Adoption Rate")
plt.legend()
plt.tight_layout()
plt.savefig("adoption_cluster_centroids.png", dpi=300)
plt.close()

# Plot a few sample trajectories per cluster
sampled = (
    pd.concat([subset, clusters], axis=1)
      .groupby("cluster")
      .head(3)
)

melted = (
    sampled.drop(columns="cluster")
      .assign(cluster=sampled["cluster"])  
      .reset_index(names="country")
      .melt(id_vars=["country", "cluster"], var_name="year", value_name="adoption_rate")
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=melted,
    x="year",
    y="adoption_rate",
    hue="cluster",
    style="country",
    estimator=None,
    alpha=0.7,
    linewidth=2,
)
plt.title("Sample Adoption Trajectories by Cluster")
plt.ylabel("Adoption Rate")
plt.xlabel("Year")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("adoption_sample_trajectories.png", dpi=300)
plt.close()

"""
Complete tslearn Example: Time Series Clustering and Classification
"""


def generate_sample_data(n_samples=100, n_timesteps=50, n_features=1):
    """Generate synthetic time series data for demonstration."""
    np.random.seed(42)
    return np.random.rand(n_samples, n_timesteps, n_features)


def demonstrate_preprocessing():
    """Demonstrate preprocessing techniques."""
    logger.info("Preprocessing Demonstration")
    
    # Generate sample data
    data = generate_sample_data(n_samples=10, n_timesteps=20)
    logger.info(f"Original data shape: {data.shape}")
    
    # Resampling
    resampler = TimeSeriesResampler(sz=30)
    resampled = resampler.fit_transform(data)
    logger.info(f"After resampling to 30 timesteps: {resampled.shape}")
    
    # Scaling
    scaler = TimeSeriesScalerMeanVariance()
    scaled = scaler.fit_transform(data)
    logger.info(f"After scaling: {scaled.shape}")
    logger.info(f"Mean of scaled data: {scaled.mean():.4f}")
    logger.info(f"Std of scaled data: {scaled.std():.4f}")


def demonstrate_clustering(plot: bool = False):
    """Demonstrate clustering techniques."""
    logger.info("=== Clustering Demonstration ===")
    
    X = generate_sample_data(n_samples=50, n_timesteps=40)
    
    # K-Shape clustering
    kshape_model = KShape(n_clusters=3, random_state=42)
    kshape_labels = kshape_model.fit_predict(X)
    logger.info(f"K-Shape clusters: {np.bincount(kshape_labels)}")
    
    # DTW-based K-Means
    dtw_model = TimeSeriesKMeans(
        n_clusters=3,
        metric="dtw",
        random_state=42,
        n_init=5
    )
    dtw_labels = dtw_model.fit_predict(X)
    logger.info(f"DTW K-Means clusters: {np.bincount(dtw_labels)}")
    
    # Visualize centroids
    if plot:
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        for idx, center in enumerate(kshape_model.cluster_centers_):
            plt.plot(center.ravel(), label=f"Cluster {idx}", linewidth=2)
        plt.title("K-Shape Centroids")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.subplot(1, 2, 2)
        for idx, center in enumerate(dtw_model.cluster_centers_):
            plt.plot(center.ravel(), label=f"Cluster {idx}", linewidth=2)
        plt.title("DTW K-Means Centroids")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig("clustering_centroids.png", dpi=300)
        plt.close()
    logger.info("Saved clustering visualization to 'clustering_centroids.png'")


def demonstrate_classification():
    """Demonstrate classification with proper cross-validation."""
    logger.info("=== Classification Demonstration ===")
    
    X = generate_sample_data(n_samples=200, n_timesteps=50)
    y = np.random.randint(0, 2, size=200)
    
    # Create pipeline with proper preprocessing
    clf = make_pipeline(
        TimeSeriesScalerMeanVariance(),
        KNeighborsTimeSeriesClassifier(n_neighbors=5, metric="dtw")
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=tscv, scoring='accuracy')
    
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Final evaluation on hold-out set
    train_idx, test_idx = list(tscv.split(X))[-1]
    clf.fit(X[train_idx], y[train_idx])
    holdout_score = clf.score(X[test_idx], y[test_idx])
    logger.info(f"Hold-out accuracy: {holdout_score:.4f}")


def demonstrate_dtw():
    """Demonstrate DTW distance calculation."""
    logger.info("=== DTW Distance Demonstration ===")
    
    ts1 = np.array([[1], [2], [3], [4], [5], [6]])
    ts2 = np.array([[2], [3], [4], [5], [6], [7]])
    
    distance = dtw(ts1, ts2)
    logger.info(f"DTW distance between ts1 and ts2: {distance:.4f}")
    
    # Compare with Euclidean distance
    euclidean_dist = np.linalg.norm(ts1.ravel() - ts2.ravel())
    logger.info(f"Euclidean distance: {euclidean_dist:.4f}")


def demonstrate_sax():
    """Demonstrate SAX feature extraction."""
    logger.info("=== SAX Feature Extraction Demonstration ===")
    
    X = generate_sample_data(n_samples=10, n_timesteps=50)
    
    sax = SymbolicAggregateApproximation(
        n_segments=10,
        alphabet_size_avg=5
    )
    X_sax = sax.fit_transform(X)
    
    logger.info(f"Original shape: {X.shape}")
    logger.info(f"SAX shape: {X_sax.shape}")
    logger.info(f"Sample SAX representation (first 2 series):")
    for i in range(2):
        sax_str = ''.join(X_sax[i].ravel().astype(str))
        logger.info(f"  Series {i}: {sax_str[:50]}...")


def main():
    """Run all demonstrations."""
    demonstrate_preprocessing()
    demonstrate_clustering()
    demonstrate_classification()
    demonstrate_dtw()
    demonstrate_sax()
    
    logger.info("=== All demonstrations completed! ===")


if __name__ == "__main__":
    main()
