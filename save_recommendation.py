from supabase import create_client
from models.collabFilteringModel import CollaborativeFilteringModel
import tensorflow as tf
import joblib
import numpy as np
from datetime import datetime

# Supabase credentials
SUPABASE_URL = "https://ympkztethjtejhdajsol.supabase.co"
SUPABASE_SERVICE_ROLE = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InltcGt6dGV0aGp0ZWpoZGFqc29sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzYyNTMxNSwiZXhwIjoyMDUzMjAxMzE1fQ.A7FFMWaTtLPNR6GAuV8pqGe6Hs5Z0iQabcqRQqg8mZc"
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

# Load model
user_encoder = joblib.load("api/user_encoder.pkl")
item_encoder = joblib.load("api/item_encoder.pkl")

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

model = CollaborativeFilteringModel(num_users, num_items)
model([tf.constant([0]), tf.constant([0])])
model.load_weights("api/my_collab_model.weights.h5")

# Fetch all user IDs with both auth_id and id
users = supabase.table("profiles").select("auth_id,id").execute().data

# Filter out users with None auth_id
success_count = 0
skipped_count = 0

for user in users:
    uid = user["id"]
    user_id = user["id"]

    if uid not in user_encoder.classes_:
        print(f"Skipping user {uid}: Not in training data")
        skipped_count += 1
        continue

    try:
        user_idx = user_encoder.transform([uid])[0]

        item_ids = np.arange(num_items)
        user_batch = np.full_like(item_ids, user_idx)
        scores = model.predict([user_batch, item_ids], verbose=0).flatten()
        top_ids = np.argsort(scores)[::-1][:5]
        recs = item_encoder.inverse_transform(top_ids).tolist()

        response = supabase.table("recommendations").upsert({
            "user_id": user_id,
            "items": recs,
            "updated_at": datetime.utcnow().isoformat()
        }).execute()

        if 'error' in response:
            print(f"Error saving recommendations for user {uid}: {response['error']}")
        else:
            success_count += 1
    except Exception as e:
        print(f"Error processing user {uid}: {str(e)}")

print(f"Recommendations updated for {success_count} users. Skipped {skipped_count} users.")