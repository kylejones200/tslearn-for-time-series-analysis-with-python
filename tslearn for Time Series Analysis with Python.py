"""Generated from Jupyter notebook: tslearn for Time Series Analysis with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import numpy as np


def main():
    X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    print(X.shape)
    from tslearn.preprocessing import TimeSeriesResampler

    resampler = TimeSeriesResampler(sz=10)
    X_resampled = resampler.fit_transform(X)
    print(X_resampled.shape)
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled)
    import matplotlib.pyplot as plt
    from tslearn.clustering import KShape

    X = np.random.rand(100, 50, 1)
    kshape = KShape(n_clusters=3, random_state=0)
    y_pred = kshape.fit_predict(X)
    for centroid in kshape.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("K-Shape Cluster Centroids")
    plt.show()
    from tslearn.clustering import TimeSeriesKMeans

    dtw_kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
    y_pred = dtw_kmeans.fit_predict(X)
    for centroid in dtw_kmeans.cluster_centers_:
        plt.plot(centroid.ravel())
    plt.title("DTW K-Means Cluster Centroids")
    plt.show()
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    X = np.random.rand(200, 50, 1)
    y = np.random.randint(0, 2, 200)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    from tslearn.metrics import dtw

    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    distance = dtw(ts1, ts2)
    print(f"DTW Distance: {distance:.2f}")
    from tslearn.piecewise import SymbolicAggregateApproximation

    sax = SymbolicAggregateApproximation(n_segments=5, alphabet_size_avg=3)
    X_sax = sax.fit_transform(X)
    print(X_sax)


def main() -> None:
    main()


if __name__ == "__main__":
    main()
