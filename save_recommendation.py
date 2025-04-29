from __future__ import annotations
from datetime import datetime
import numpy as np
import joblib
import tensorflow as tf
from supabase import create_client
from models.collabFilteringModel import CollaborativeFilteringModel
from models.knn_cluster_model import KNNRecommender

SUPABASE_URL = "https://ympkztethjtejhdajsol.supabase.co"
SUPABASE_SERVICE_ROLE = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InltcGt6dGV"
    "0aGp0ZWpoZGFqc29sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzYyNTMxNSwiZXhwIjoy"
    "MDUzMjAxMzE1fQ.A7FFMWaTtLPNR6GAuV8pqGe6Hs5Z0iQabcqRQqg8mZc"
)

auth_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

# Label encoders for CF model
user_encoder = joblib.load("api/user_encoder.pkl")
item_encoder = joblib.load("api/item_encoder.pkl")
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

# Collaborative Filtering model (TensorFlow)
cf_model = CollaborativeFilteringModel(num_users, num_items)
cf_model([tf.constant([0]), tf.constant([0])])
cf_model.load_weights("api/my_collab_model.weights.h5")

# KNN recommender
knn_model = KNNRecommender()
knn_model.load("api/knn")

def top5_cf(user_id: str) -> list[str]:
    if user_id not in user_encoder.classes_:
        return []
    uid_idx = user_encoder.transform([user_id])[0]
    scores = cf_model.predict(
        [np.full(num_items, uid_idx), np.arange(num_items)], verbose=0
    ).flatten()
    top_indices = scores.argsort()[::-1][:5]
    return item_encoder.inverse_transform(top_indices).tolist()


def top5_knn(user_id: str) -> list[str]:
    return knn_model.recommend(user_id, k=5)

def process_all_users():
    users = auth_client.table("profiles").select("id").execute().data

    success, skipped = 0, 0
    for user in users:
        uid = user["id"]

        cf_recs = top5_cf(uid)
        knn_recs = top5_knn(uid)

        combined = cf_recs + [item for item in knn_recs if item not in cf_recs]
        combined = combined[:10]

        if not combined:
            skipped += 1
            continue

        auth_client.table("recommendations").upsert(
            {
                "user_id": uid,
                "items": combined,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).execute()
        success += 1

    print(
        f"Recommendations updated for {success} users. Skipped {skipped} users."
    )

if __name__ == "__main__":
    process_all_users()
