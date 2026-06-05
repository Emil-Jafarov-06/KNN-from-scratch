import numpy as np


class KNN:
    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        weights: str = "uniform",
    ) -> None:
        self.y = None
        self.X = None
        if type(k) != int or k < 1:
            raise ValueError("k must be greater than 0!")
        self.k = k
        if metric not in ["euclidean", "manhattan", "minkowski"]:
            raise ValueError("Invalid distance metric!")
        self.metric = metric
        if (q is None or type(q) not in [int, float] or q <= 0) and metric == "minkowski":
            raise ValueError("q must be a positive numeric value!")
        self.q = q
        if task not in ["classification", "regression"]:
            raise ValueError("Invalid task type!")
        self.task = task
        if weights not in ["uniform", "distance"]:
            raise ValueError("Invalid weights type!")
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """ Memorize the training data . Returns self . """
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("X and y must be numpy arrays!")
        if X.ndim != 2:
            raise ValueError("Feature should be a 2D array.")
        if y.ndim != 1:
            raise ValueError("Target should be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Feature and target data should have the same number of rows.")
        if X.shape[0] < self.k:
            raise ValueError("Number of rows should be greater than or equal to k.")

        self.X = X.copy()
        self.y = y.copy()

        return self



    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict labels ( classification ) or values ( regression )."""
        if self.weights == "uniform":
            distances = self.__compute_distance(X)
            values = self.__find_neighbor_values(distances)
            if self.task == "classification":
                classes = np.unique(self.y)

                comparisons = values[:, :, None] == classes[None, None, :]
                votes = np.sum(comparisons, axis=1)

                winning_class_indices = np.argmax(votes, axis=1)
                predictions = classes[winning_class_indices]
                return predictions

            elif self.task == "regression":
                averages = np.mean(values, axis = 1)
                return averages

        elif self.weights == "distance":
            distances = self.__compute_distance(X)
            values, neighbor_distances = self.__find_neighbors_for_weighted(distances)
            neighbor_weights = 1 / (neighbor_distances + 1e-8)
            if self.task == "classification":
                classes = np.unique(self.y)

                comparisons = values[:, :, None] == classes[None, None, :]
                weighted_votes = np.sum(comparisons * neighbor_weights[:, :, None],axis=1)

                winning_class_indices = np.argmax(weighted_votes, axis=1)
                return classes[winning_class_indices]

            elif self.task == "regression":
                weighted_sum = np.sum(values * neighbor_weights,axis=1,)
                total_weight = np.sum(neighbor_weights,axis=1)
                return weighted_sum / total_weight


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Return per - class probabilities ( classification only ).
    Output shape : ( n_queries , n_classes )."""
        if self.task == "regression":
            raise ValueError("This method is not compatible with regression task!")
        distances = self.__compute_distance(X)
        classes = np.unique(self.y)
        if self.weights == "uniform":
            values = self.__find_neighbor_values(distances)

            comparisons = values[:, :, None] == classes[None, None, :]
            votes = np.sum(comparisons, axis=1)
            probabilities = votes / np.sum(votes, axis=1, keepdims=True)
            return probabilities

        elif self.weights == "distance":
            values, neighbor_distances = self.__find_neighbors_for_weighted(distances)
            neighbor_weights = 1 / (neighbor_distances + 1e-8)

            comparisons = values[:, :, None] == classes[None, None, :]
            weighted_votes = np.sum(comparisons * neighbor_weights[:, :, None],axis=1)
            probabilities = weighted_votes / np.sum(weighted_votes, axis=1, keepdims=True)
            return probabilities



    def __compute_distance(self, X: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise ValueError("The model is not fitted yet!")

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array!")

        if X.ndim != 2:
            raise ValueError("Query features should be a 2D array.")

        if X.shape[1] != self.X.shape[1]:
            raise ValueError("Training and query data must have the same number of features.")

        X_query = X[:, None, :]
        X_training = self.X[None, :, :]

        differences = X_query - X_training

        if self.metric == "euclidean":
            distances = np.sqrt(np.sum(np.square(differences), axis=2))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(differences), axis=2)
        elif self.metric == "minkowski":
            distances = np.sum(np.abs(differences) ** self.q, axis=2) ** (1 / self.q)

        return distances

    def __find_neighbor_values(self, distances: np.ndarray) -> np.ndarray:
        nearest_neighbor_indices = np.argsort(distances, axis = 1)[:, 0:self.k]
        return self.y[nearest_neighbor_indices]

    def __find_neighbors_for_weighted(
            self,
            distances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_values = self.y[nearest_neighbor_indices]
        neighbor_distances = np.take_along_axis(distances, nearest_neighbor_indices, axis=1)
        return neighbor_values, neighbor_distances