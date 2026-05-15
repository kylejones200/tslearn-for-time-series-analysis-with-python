# Description: Short example for resampled life expectancy tslearn.


import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_io import read_csv
from tslearn.clustering import TimeSeriesKMeans
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    data_path = Path(
        "/Users/kylejonespatricia/time_series/WIP/utilities/data/Hierarchical_Amtrak_Ridership.csv"
    )
    df = read_csv(data_path)
    logger.info(df.head())

    pivot = (
        df.groupby(["station_id", "year"])["Ridership"]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index(axis="columns")
    )

    # keep stations with a reasonable amount of history
    pivot = pivot.loc[pivot.count(axis=1) >= 10]

    # forward/backward fill small gaps, then zero-fill what remains
    pivot = pivot.ffill(axis=1).bfill(axis=1).fillna(0.0)

    X = pivot.to_numpy()[:, :, None]


    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    # unify length across stations (e.g., 20 year samples)
    resampler = TimeSeriesResampler(sz=20)
    X_resampled = resampler.fit_transform(X_scaled)


    model = TimeSeriesKMeans(n_clusters=6, metric="dtw", random_state=42, n_init=4)
    labels = model.fit_predict(X_resampled)
    clusters = pd.Series(labels, index=pivot.index, name="cluster")
    logger.info(clusters.value_counts())


    plt.figure(figsize=(10, 5))
    for idx, center in enumerate(model.cluster_centers_):
        plt.plot(center.ravel(), label=f"Cluster {idx}")
    plt.title("Amtrak Ridership Trajectory Centroids")
    plt.xlabel("Resampled Year Index")
    plt.ylabel("Scaled Ridership")
    plt.legend()
    plt.tight_layout()
    plt.savefig("amtrak_cluster_centroids.png", dpi=300)
    plt.close()


    sampled = pd.concat([pivot, clusters], axis=1).groupby("cluster").head(5)

    melted = (
        sampled.drop(columns="cluster")
        .assign(cluster=sampled["cluster"].values)
        .reset_index(names="station_id")
        .melt(id_vars=["station_id", "cluster"], var_name="year", value_name="ridership")
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=melted,
        x="year",
        y="ridership",
        hue="cluster",
        style="station_id",
        estimator=None,
        alpha=0.6,
    )
    plt.title("Representative Station Trajectories")
    plt.ylabel("Ridership (boardings + alightings)")
    plt.tight_layout()
    plt.savefig("amtrak_sample_trajectories.png", dpi=300)
    plt.close()


    sax = SymbolicAggregateApproximation(n_segments=8, alphabet_size_avg=5)
    X_sax = sax.fit_transform(X_resampled)
    sax_strings = pd.Series(
        ["".join(seq) for seq in X_sax.reshape(len(X_sax), -1)], index=pivot.index
    )
    logger.info(sax_strings.head())


if __name__ == "__main__":
    main()
