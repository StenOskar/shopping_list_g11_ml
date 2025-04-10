from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np
import tensorflow as tf
from models.collabFilteringModel import CollaborativeFilteringModel

app = FastAPI()

user_encoder = joblib.load("user_encoder.pkl")
item_encoder = joblib.load("item_encoder.pkl")

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)

# Build the model by calling it with a tiny sample
dummy_user = tf.constant([0], dtype=tf.int32)
dummy_item = tf.constant([0], dtype=tf.int32)
_ = model([dummy_user, dummy_item])  # forward pass -> "builds" the model

model.load_weights("my_collab_model.weights.h5")

def recommend_for_user(model, raw_user_id, top_n=5):
    try:
        encoded_user_id = user_encoder.transform([raw_user_id])[0]
    except ValueError:
        raise ValueError(f"User ID '{raw_user_id}' not found in the encoder.")

    try:
        all_item_indices = np.arange(num_items)
        user_batch = np.full_like(all_item_indices, encoded_user_id)
        scores = model.predict([user_batch, all_item_indices], verbose=0).flatten()

        top_indices = np.argsort(scores)[::-1][:top_n]
        recommended_item_ids = item_encoder.inverse_transform(top_indices)
        return recommended_item_ids
    except Exception as e:
        raise RuntimeError(f"An error occurred during recommendation: {str(e)}")

@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_n: int = 5):
    results = recommend_for_user(model, user_id, top_n)
    return {"user_id": user_id, "recommendations": results.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
