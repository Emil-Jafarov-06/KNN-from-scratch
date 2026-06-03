from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.knn import KNN


def run_knn_runtime_benchmark() -> None:
    """Measure KNN prediction time as training-set size increases."""
    rng = np.random.default_rng(42)

    training_sizes = [100, 500, 1000, 5000, 10000]

    n_queries = 100
    n_features = 10
    repetitions = 5

    X_query = rng.normal(size=(n_queries, n_features))

    median_times = []

    for n_training in training_sizes:
        X_train = rng.normal(size=(n_training, n_features))
        y_train = rng.integers(low=0, high=2, size=(n_training,))

        model = KNN(
            k=5,
            metric="euclidean",
            task="classification",
            weights="uniform",
        )

        model.fit(X_train, y_train)
        run_times = []

        for _ in range(repetitions):
            start = perf_counter()
            model.predict(X_query)
            end = perf_counter()
            run_times.append(end - start)

        median_times.append(np.median(run_times))

    median_times = np.array(median_times)

    # Estimate slope of log(time) versus log(N).
    exponent, intercept = np.polyfit(
        np.log(training_sizes),
        np.log(median_times),
        deg=1,
    )

    print("Training sizes:", training_sizes)
    print("Median prediction times:", median_times)
    print(f"Estimated scaling exponent: {exponent:.3f}")

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.loglog(
        training_sizes,
        median_times,
        marker="o",
        label="Measured prediction time",
    )

    plt.xlabel("Training-set size N")
    plt.ylabel("Prediction time for 100 queries (seconds)")
    plt.title("KNN Prediction Runtime vs. Training-Set Size")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir / "knn_runtime_benchmark.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

if __name__ == "__main__":
    run_knn_runtime_benchmark()