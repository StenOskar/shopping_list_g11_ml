from services.data_loader import SupabaseDataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
from models.collabFilteringModel import CollaborativeFilteringModel
from models.knn_cluster_model import KNNRecommender

# Supabase credentials
SUPABASE_URL = "https://ympkztethjtejhdajsol.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InltcGt6dGV0aGp0ZWpoZGFqc29sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc2MjUzMTUsImV4cCI6MjA1MzIwMTMxNX0.GA21O-DEqkNCO1DbVEJ3KHh74fg5e0ZxejNnFrwhHto"

# Fetch purchase data from Supabase
loader = SupabaseDataLoader(SUPABASE_URL, SUPABASE_KEY)
df_receipts = loader.fetch_receipts()
df_items = loader.fetch_receipt_items()

# Prepare data
df_receipts.rename(columns={'id': 'receipt_id'}, inplace=True)
df_items.rename(columns={'name': 'item_id'}, inplace=True)
df_merged = pd.merge(df_receipts, df_items, on='receipt_id')

knn = KNNRecommender(n_neighbors=20)
knn.fit(df_merged[["user_id", "item_id"]])
knn.save("api/knn")
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df_merged["user_id_encoded"] = user_encoder.fit_transform(df_merged["user_id"])
df_merged["item_id_encoded"] = item_encoder.fit_transform(df_merged["item_id"])

num_users = df_merged["user_id_encoded"].nunique()
num_items = df_merged["item_id_encoded"].nunique()

df_merged["label"] = 1.0

df_merged = df_merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Build and train the model
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(
    x=[
        df_merged["user_id_encoded"].values,
        df_merged["item_id_encoded"].values
    ],
    y=df_merged["label"].values,
    epochs=5,
    batch_size=128
)

# Save the trained weights
model.save_weights("api/my_collab_model.weights.h5")
print("Model training finished and weights saved.")

# Save the label encoders for later use in server.py
joblib.dump(user_encoder, "api/user_encoder.pkl")
joblib.dump(item_encoder, "api/item_encoder.pkl")
print("Encoders saved as api/user_encoder.pkl and api/item_encoder.pkl")