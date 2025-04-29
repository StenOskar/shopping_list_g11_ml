import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

class KNNRecommender:
    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.knn_model: NearestNeighbors | None = None
        self.user_item_matrix: sp.csr_matrix | None = None
        self.user_encoder: LabelEncoder | None = None
        self.item_encoder: LabelEncoder | None = None

    def fit(self, df, user_col: str = "user_id", item_col: str = "item_id"):
        self.user_encoder = LabelEncoder().fit(df[user_col])
        self.item_encoder = LabelEncoder().fit(df[item_col])

        user_idx = self.user_encoder.transform(df[user_col])
        item_idx = self.item_encoder.transform(df[item_col])

        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        self.user_item_matrix = sp.csr_matrix(
            (np.ones_like(user_idx, dtype=np.float32),
             (user_idx, item_idx)),
            shape=(n_users, n_items),
        )

        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric,
            algorithm="brute",
        ).fit(self.user_item_matrix)

    def recommend(self, user_raw_id: str, k: int = 5) -> list[str]:

        if self.knn_model is None:
            raise ValueError("Model not trained. Call fit() or load() first.")

        if user_raw_id not in self.user_encoder.classes_:
            return []

        u_idx = self.user_encoder.transform([user_raw_id])[0]

        max_neigh = min(self.n_neighbors + 1, self.user_item_matrix.shape[0])

        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix[u_idx],
            n_neighbors=max_neigh,
            return_distance=True
        )

        neighbour_idxs = indices.flatten()
        if neighbour_idxs.size == 0:
            return []

        neighbour_idxs = neighbour_idxs[1:]

        neighbour_matrix = self.user_item_matrix[neighbour_idxs]
        scores = np.asarray(neighbour_matrix.sum(axis=0)).flatten()

        seen = set(self.user_item_matrix[u_idx].indices)
        candidate_order = np.argsort(scores)[::-1]        # high → low
        rec_idxs = [idx for idx in candidate_order if idx not in seen][:k]
        return self.item_encoder.inverse_transform(rec_idxs).tolist()


    def save(self, path_prefix: str):
        joblib.dump(self.knn_model, f"{path_prefix}_knn.pkl")
        joblib.dump(self.user_encoder, f"{path_prefix}_user_enc.pkl")
        joblib.dump(self.item_encoder, f"{path_prefix}_item_enc.pkl")

    def load(self, path_prefix: str):
        self.knn_model   = joblib.load(f"{path_prefix}_knn.pkl")
        self.user_encoder = joblib.load(f"{path_prefix}_user_enc.pkl")
        self.item_encoder = joblib.load(f"{path_prefix}_item_enc.pkl")
        self.user_item_matrix = self.knn_model._fit_X

