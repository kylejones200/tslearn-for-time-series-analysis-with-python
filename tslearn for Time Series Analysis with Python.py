"""Generated from Jupyter notebook: tslearn for Time Series Analysis with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# !pip install tslearn  # Jupyter-only


# --- code cell ---

import numpy as np


def main():
    # Create a synthetic univariate time series dataset
    X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    print(X.shape)  # (3 samples, 5 timesteps)

    from tslearn.preprocessing import TimeSeriesResampler

    # Resample to 10 timesteps
    resampler = TimeSeriesResampler(sz=10)
    X_resampled = resampler.fit_transform(X)
    print(X_resampled.shape)  # (3 samples, 10 timesteps)

    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled)

    import matplotlib.pyplot as plt
    from tslearn.clustering import KShape

    # Synthetic time series dataset
    X = np.random.rand(100, 50, 1)  # 100 samples, 50 timesteps

    # Apply K-Shape clustering
    kshape = KShape(n_clusters=3, random_state=0)
    y_pred = kshape.fit_predict(X)

    # Plot cluster centroids
    for centroid in kshape.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("K-Shape Cluster Centroids")
    plt.show()

    from tslearn.clustering import TimeSeriesKMeans

    # Apply DTW-based K-Means
    dtw_kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
    y_pred = dtw_kmeans.fit_predict(X)

    # Plot cluster centroids
    for centroid in dtw_kmeans.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("DTW K-Means Cluster Centroids")
    plt.show()

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    # Create synthetic dataset
    X = np.random.rand(200, 50, 1)  # 200 samples, 50 timesteps
    y = np.random.randint(0, 2, 200)  # Binary labels

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a k-NN classifier with DTW
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    from tslearn.metrics import dtw

    # Two synthetic time series
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])

    # Compute DTW distance
    distance = dtw(ts1, ts2)
    print(f"DTW Distance: {distance:.2f}")

    from tslearn.piecewise import SymbolicAggregateApproximation

    # Apply SAX
    sax = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=3)
    X_sax = sax.fit_transform(X)
    print(X_sax)


if __name__ == "__main__":
    main()
