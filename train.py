import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models.collabFilteringModel import CollaborativeFilteringModel
from models.knn_cluster_model import KNNRecommender
from services.data_loader import SupabaseDataLoader

# Supabase credentials
SUPABASE_URL = "https://ympkztethjtejhdajsol.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InltcGt6dGV0aGp0ZWpoZGFqc29sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc2MjUzMTUsImV4cCI6MjA1MzIwMTMxNX0.GA21O-DEqkNCO1DbVEJ3KHh74fg5e0ZxejNnFrwhHto"

# Fetch data from Supabase
print("Fetching data from Supabase...")
loader = SupabaseDataLoader(SUPABASE_URL, SUPABASE_KEY)
df_receipts = loader.fetch_receipts()
df_items = loader.fetch_receipt_items()

# Prepare data
print("Preparing data...")
df_receipts.rename(columns={'id': 'receipt_id'}, inplace=True)
df_items.rename(columns={'name': 'item_id'}, inplace=True)
df_merged = pd.merge(df_receipts, df_items, on='receipt_id')

# Create label encoders
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit encoders on all users and items
user_encoder.fit(df_merged["user_id"].unique())
item_encoder.fit(df_merged["item_id"].unique())

num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
print(f"Number of users: {num_users}, Number of items: {num_items}")

# Generate negative samples
print("Generating negative samples...")
all_items = df_merged['item_id'].unique()
negative_samples = []
neg_sample_ratio = 2

for user in df_merged['user_id'].unique():
    # Get items this user has interacted with
    user_items = set(df_merged[df_merged['user_id'] == user]['item_id'])

    # Find items the user hasn't interacted with
    negative_items = list(set(all_items) - user_items)

    # Limit number of negative samples per user
    num_neg_samples = min(len(negative_items), len(user_items) * neg_sample_ratio)
    if num_neg_samples > 0:
        np.random.shuffle(negative_items)
        selected_neg_items = negative_items[:num_neg_samples]

        # Create negative samples
        for item in selected_neg_items:
            negative_samples.append({
                'user_id': user,
                'item_id': item,
                'label': 0.0
            })

# Create combined dataset with positive and negative samples
df_positive = df_merged[['user_id', 'item_id']].drop_duplicates().assign(label=1.0)
df_negative = pd.DataFrame(negative_samples)
df_combined = pd.concat([df_positive, df_negative], ignore_index=True)

# Shuffle the data
df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Split into train and validation sets
print("Splitting data into train and validation sets...")
train_df, val_df = train_test_split(
    df_combined, test_size=0.2, random_state=42, stratify=df_combined['label']
)

# Encode user and item IDs
train_df['user_id_encoded'] = user_encoder.transform(train_df['user_id'])
train_df['item_id_encoded'] = item_encoder.transform(train_df['item_id'])
val_df['user_id_encoded'] = user_encoder.transform(val_df['user_id'])
val_df['item_id_encoded'] = item_encoder.transform(val_df['item_id'])

# Train KNN model (using only positive interactions)
print("Training KNN model...")
knn = KNNRecommender(n_neighbors=20)
knn.fit(df_positive[["user_id", "item_id"]])
knn.save("api/knn")
print("KNN model saved")

# Build and train collaborative filtering model
print("Training collaborative filtering model...")
model = CollaborativeFilteringModel(num_users, num_items, embedding_dim=50)

# Configure model training
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=3,
    mode='max',
    restore_best_weights=True
)

# Train model with validation data and early stopping
history = model.fit(
    x=[
        train_df["user_id_encoded"].values,
        train_df["item_id_encoded"].values
    ],
    y=train_df["label"].values,
    validation_data=(
        [val_df["user_id_encoded"].values, val_df["item_id_encoded"].values],
        val_df["label"].values
    ),
    epochs=15,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)

# Save model weights and encoders
print("Saving collaborative filtering model...")
model.save_weights("api/my_collab_model.weights.h5")
joblib.dump(user_encoder, "api/user_encoder.pkl")
joblib.dump(item_encoder, "api/item_encoder.pkl")

print("Training complete! Models and encoders saved.")