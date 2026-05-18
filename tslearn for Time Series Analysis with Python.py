"""Generated from Jupyter notebook: tslearn for Time Series Analysis with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance


def create_a_synthetic_univariate_time_series_datase() -> None:
    X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    print(X.shape)
    resampler = TimeSeriesResampler(sz=10)
    X_resampled = resampler.fit_transform(X)
    print(X_resampled.shape)
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled)
    X = np.random.rand(100, 50, 1)
    kshape = KShape(n_clusters=3, random_state=0)
    y_pred = kshape.fit_predict(X)
    for centroid in kshape.cluster_centers_:
        plt.plot(centroid.ravel())

    plt.title("K-Shape Cluster Centroids")
    plt.show()
    dtw_kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
    y_pred = dtw_kmeans.fit_predict(X)
    for centroid in dtw_kmeans.cluster_centers_:
        plt.plot(centroid.ravel())

    plt.title("DTW K-Means Cluster Centroids")
    plt.show()
    X = np.random.rand(200, 50, 1)
    y = np.random.randint(0, 2, 200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    distance = dtw(ts1, ts2)
    print(f"DTW Distance: {distance:.2f}")
    sax = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=3)
    X_sax = sax.fit_transform(X)
    print(X_sax)


def main() -> None:
    create_a_synthetic_univariate_time_series_datase()


if __name__ == "__main__":
    main()
