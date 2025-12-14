# classifier.py
import numpy as np
from collections import Counter

class KNNClassifier:
    """

    - fit(X, y): store training data
    - predict(x): predict label for a single sample
    - predict_batch(X): predict for batch
    - load_training_csv(path, delimiter=',') -> (X, y)

    The training CSV is expected to have features in the first columns and the label in the last column.
    Labels can be numeric or strings. Missing/NaN features are filled with 0.
    """

    def __init__(self, k=3):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.X_train = X
        self.y_train = y

    def _check_trained(self):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Classifier not trained. Call fit(X, y) first.")

    def _ensure_feature_length(self, x):
        x = np.array(x, dtype=float).ravel()
        n_train = self.X_train.shape[1]

        if x.size > n_train:
            return x[:n_train]
        elif x.size < n_train:
            padded = np.zeros(n_train, dtype=float)
            padded[:x.size] = x
            return padded
        return x

    def predict(self, x):
        self._check_trained()
        x = self._ensure_feature_length(x)

        # Euclidean distances
        dists = np.linalg.norm(self.X_train - x, axis=1)
        idx = np.argsort(dists)[:self.k]

        neighbor_labels = self.y_train[idx]
        neighbor_dists = dists[idx]

        # Majority voting
        votes = Counter(neighbor_labels)
        top_votes = votes.most_common()
        best_label = top_votes[0][0]

        # Tie-breaking by lowest distance
        if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
            tied = [lab for lab, cnt in top_votes if cnt == top_votes[0][1]]
            avg_dists = {
                t: neighbor_dists[neighbor_labels == t].mean()
                for t in tied
            }
            best_label = min(avg_dists, key=avg_dists.get)

        return best_label, dict(votes), list(neighbor_labels), list(neighbor_dists)

    def predict_batch(self, X):
        return [self.predict(x)[0] for x in X]

    @staticmethod
    def load_training_csv(path, delimiter=','):
        data = []
        labels = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = [p.strip() for p in line.split(delimiter) if p.strip() != '']
                if len(parts) < 2:
                    continue

                *feat_parts, lab = parts
                feats = []

                for p in feat_parts:
                    try:
                        feats.append(float(p))
                    except:
                        clean = p.replace(",", "")
                        try:
                            feats.append(float(clean))
                        except:
                            feats.append(0.0)

                data.append(feats)
                labels.append(lab)

        maxlen = max(len(r) for r in data)
        X = np.zeros((len(data), maxlen), dtype=float)

        for i, r in enumerate(data):
            X[i, :len(r)] = r

        y = np.array(labels)
        return X, y



