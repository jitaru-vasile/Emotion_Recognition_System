import numpy as np
from distance import euclidean


class KNearestNeighbors:

    def __init__(self, k=3, distance_metric=euclidean):
        """Initialize k value and distance metric used for model."""
        self.k = k
        self.distance = distance_metric
        self.data = None

    def train(self, X, y):
        """Zip labels and input data together for classification."""
        # raise value error if inputs are wrong length or different types
        if len(X) != len(y) or type(X) != type(y):
            raise ValueError("X and y are incompatible.")
        # convert ndarrays to lists
        if type(X) == np.ndarray:
            X, y = X.tolist(), y.tolist()
        # set data attribute containing instances and labels
        self.data = [X[i]+[y[i]] for i in range(len(X))]

    def predict(self, array):
        y_test = np.array([], dtype=int)
        for a in array:
            """Predict class based on k-nearest neighbors."""
            neighbors = []
            # create mapping from distance to instance
            distances = {self.distance(x[:-1], a): x for x in self.data}
            # collect classes of k instances with shortest distance
            for key in sorted(distances.keys())[:self.k]:
                neighbors.append(distances[key][-1])
            # return most common vote
            y_test = np.append(y_test, max(set(neighbors), key=neighbors.count))
        return y_test

    def score(self, X_test, y_test):
        prediction = self.predict(X_test)
        scoreInd = 0
        for x in range(0, np.size(prediction), 1):
            if prediction[x] == y_test[x]:
                scoreInd = scoreInd + 1
        return scoreInd / np.size(prediction)