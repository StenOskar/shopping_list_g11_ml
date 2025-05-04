import os
from dotenv import load_dotenv
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from item_filters import should_exclude_item
from models.collabFilteringModel import CollaborativeFilteringModel
from models.knn_cluster_model import KNNRecommender
from services.data_loader import SupabaseDataLoader

# Load environment variables
load_dotenv('secrets.env')

# Fetch data from Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials; set SUPABASE_URL and SUPABASE_SERVICE_ROLE")

print("Fetching data from Supabase …")
loader = SupabaseDataLoader(SUPABASE_URL, SUPABASE_KEY)
df_receipts = loader.fetch_receipts()
df_items = loader.fetch_receipt_items()

# Create label encoders
df_receipts.rename(columns={"id": "receipt_id"}, inplace=True)
df_items.rename(columns={"name": "item_id"}, inplace=True)

df_merged = pd.merge(df_receipts, df_items, on="receipt_id")

# Filter out stoplist items
print("Filtering out stoplist items...")
initial_item_count = len(df_items)
df_items = df_items[~df_items["item_id"].apply(should_exclude_item)]
df_merged = pd.merge(df_receipts, df_items, on="receipt_id")
filtered_count = initial_item_count - len(df_items)
print(f"Filtered out {filtered_count} stoplist items")

# Fit encoders on all users and items
user_encoder = LabelEncoder().fit(df_merged["user_id"].unique())
item_encoder = LabelEncoder().fit(df_merged["item_id"].unique())

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
print(f"Number of users: {num_users:,}  •  Number of items: {num_items:,}")

# Create negative samples
neg_ratio = 2
rng = np.random.default_rng(42)

# Create sparse matrix for interactions
rows = user_encoder.transform(df_merged["user_id"])
cols = item_encoder.transform(df_merged["item_id"])
interaction = sp.csr_matrix(
    (np.ones_like(rows, dtype=bool), (rows, cols)), shape=(num_users, num_items)
)

neg_users, neg_items = [], []
all_items = np.arange(num_items)

# Sample negative items for each user
for u in range(num_users):
    pos = interaction[u].indices
    available = np.setdiff1d(all_items, pos, assume_unique=True)
    desired_neg = len(pos) * neg_ratio
    if desired_neg == 0 or len(available) == 0:
        continue
    replace_flag = len(available) < desired_neg
    sample = rng.choice(available, desired_neg, replace=replace_flag)

    neg_users.append(np.full(desired_neg, u, dtype=np.int32))
    neg_items.append(sample)

neg_users = np.concatenate(neg_users)
neg_items = np.concatenate(neg_items)

neg_users_raw = user_encoder.inverse_transform(neg_users)
neg_items_raw = item_encoder.inverse_transform(neg_items)

df_negative = pd.DataFrame({
    "user_id": neg_users_raw,
    "item_id": neg_items_raw,
    "label": 0.0,
})

df_positive = (
    df_merged[["user_id", "item_id"]]
    .drop_duplicates()
    .assign(label=1.0)
)

# Encode user and item IDs
print("Building training dataframe …")
df_combined = pd.concat([df_positive, df_negative], ignore_index=True)
df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

train_df, val_df = train_test_split(
    df_combined,
    test_size=0.2,
    stratify=df_combined["label"],
    random_state=42,
)

# Encode user and item IDs
train_df['user_id_encoded'] = user_encoder.transform(train_df['user_id'])
train_df['item_id_encoded'] = item_encoder.transform(train_df['item_id'])
val_df['user_id_encoded'] = user_encoder.transform(val_df['user_id'])
val_df['item_id_encoded'] = item_encoder.transform(val_df['item_id'])

print("Training KNN recommender …")
knn = KNNRecommender(n_neighbors=20)
knn.fit(df_positive[["user_id", "item_id"]])
knn.save("api/knn")
print("KNN model saved")

print("Training collaborative‑filtering model …")
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=3, mode="max", restore_best_weights=True
)

# Train model with validation data and early stopping
history = model.fit(
    x=[
        train_df["user_id_encoded"].values,
        train_df["item_id_encoded"].values,
    ],
    y=train_df["label"].values,
    validation_data=(
        [val_df["user_id_encoded"].values, val_df["item_id_encoded"].values],
        val_df["label"].values,
    ),
    epochs=15,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1,
)

# Save model weights and encoders

print("Saving artefacts …")
MODEL_DIR = "api"
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_weights(os.path.join(MODEL_DIR, "my_collab_model.weights.h5"))
joblib.dump(user_encoder, os.path.join(MODEL_DIR, "user_encoder.pkl"))
joblib.dump(item_encoder, os.path.join(MODEL_DIR, "item_encoder.pkl"))

print("Training complete!  ({} negatives per positive, finished at {})".format(
    neg_ratio, datetime.utcnow().isoformat()
))
