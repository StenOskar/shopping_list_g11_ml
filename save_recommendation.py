from __future__ import annotations
from datetime import datetime
import os
import logging
import numpy as np
import joblib
import tensorflow as tf
from supabase import create_client, Client

from item_filters import should_exclude_item
from models.collabFilteringModel import CollaborativeFilteringModel
from models.knn_cluster_model import KNNRecommender
from dotenv import load_dotenv

# Load environment variables
load_dotenv('secrets.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")

# Global variables with proper initialization
auth_client = None
user_encoder = None
item_encoder = None
cf_model = None
knn_model = None
num_users = 0
num_items = 0


def initialize_client() -> Client:
    """Initialize and return Supabase client."""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
            raise ValueError("Missing Supabase credentials")
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise


def load_models():
    """Load all recommendation models and encoders."""
    global auth_client, user_encoder, item_encoder, cf_model, knn_model, num_users, num_items

    try:
        # Initialize Supabase client
        auth_client = initialize_client()

        # Load label encoders
        try:
            user_encoder = joblib.load("api/user_encoder.pkl")
            item_encoder = joblib.load("api/item_encoder.pkl")
            num_users = len(user_encoder.classes_)
            num_items = len(item_encoder.classes_)
        except FileNotFoundError as e:
            logger.error(f"Encoder file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading encoders: {e}")
            raise

        # Load collaborative filtering model
        try:
            cf_model = CollaborativeFilteringModel(num_users, num_items)
            cf_model([tf.constant([0]), tf.constant([0])])
            cf_model.load_weights("api/my_collab_model.weights.h5")
        except (FileNotFoundError, tf.errors.NotFoundError) as e:
            logger.error(f"CF model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading CF model: {e}")
            raise

        # Load KNN recommender
        try:
            knn_model = KNNRecommender()
            knn_model.load("api/knn")
        except FileNotFoundError as e:
            logger.error(f"KNN model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading KNN model: {e}")
            raise

    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise


def top5_cf(user_id: str) -> list[str]:
    """Generate top 5 recommendations using CF model."""
    try:
        if user_id not in user_encoder.classes_:
            logger.info(f"User {user_id} not found in CF training data")
            return []

        uid_idx = user_encoder.transform([user_id])[0]
        scores = cf_model.predict(
            [np.full(num_items, uid_idx), np.arange(num_items)], verbose=0
        ).flatten()

        top_indices = scores.argsort()[::-1][:5]
        return item_encoder.inverse_transform(top_indices).tolist()
    except Exception as e:
        logger.error(f"Error generating CF recommendations for user {user_id}: {e}")
        return []  # Graceful degradation


def top5_knn(user_id: str) -> list[str]:
    """Generate top 5 recommendations using KNN model."""
    try:
        return knn_model.recommend(user_id, k=5)
    except Exception as e:
        logger.error(f"Error generating KNN recommendations for user {user_id}: {e}")
        return []


def get_popular_items(top_n: int = 10) -> list[str]:
    """Get the most popular items based on purchase frequency."""
    try:
        # Fetch all items from the database
        response = auth_client.table("receipt_items").select("name").execute()
        items_data = response.data

        # Count occurrences of each item
        item_counts = {}
        for item in items_data:
            name = item["name"]
            if not should_exclude_item(name):
                item_counts[name] = item_counts.get(name, 0) + 1

        # Sort items by count and get the top N
        popular_items = sorted(item_counts.keys(),
                               key=lambda x: item_counts[x],
                               reverse=True)[:top_n]

        return popular_items
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        # Provide fallback default items when DB query fails
        return ["Milk", "Bread", "Eggs", "Bananas", "Apples",
                "Chicken", "Rice", "Pasta", "Tomatoes", "Onions"]


def get_recommendations_for_user(user_id: str) -> list[str]:
    """Get recommendations for a user with cold start handling."""
    cf_recs = top5_cf(user_id)
    knn_recs = top5_knn(user_id)

    # Filter out unwanted items
    cf_recs = [item for item in cf_recs if not should_exclude_item(item)]
    knn_recs = [item for item in knn_recs if not should_exclude_item(item)]

    combined = cf_recs + [item for item in knn_recs if item not in cf_recs]

    # If we have too few recommendations after filtering, use popular items
    if len(combined) < 5:
        logger.info(f"Using popular items for user {user_id} after stoplist filtering")
        popular = [item for item in get_popular_items(20) if not should_exclude_item(item)]
        combined.extend([item for item in popular if item not in combined])

    return combined[:10]


def process_all_users():
    """Process all users and update their recommendations in the database."""
    if not all([auth_client, user_encoder, item_encoder, cf_model, knn_model]):
        try:
            load_models()
        except Exception as e:
            logger.error(f"Cannot process users: {e}")
            return

    try:
        users_response = auth_client.table("profiles").select("id").execute()
        users = users_response.data
    except Exception as e:
        logger.error(f"Failed to fetch users: {e}")
        return

    popular_items = get_popular_items(10)

    success, skipped = 0, 0
    for user in users:
        try:
            uid = user["id"]

            combined = get_recommendations_for_user(uid)

            # Use cached popular items if no recommendations were found
            if not combined:
                logger.info(f"Using cached popular items for user {uid}")
                combined = popular_items

            try:
                auth_client.table("recommendations").upsert(
                    {
                        "user_id": uid,
                        "items": combined,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                ).execute()
                success += 1
            except Exception as e:
                logger.error(f"Failed to update recommendations for user {uid}: {e}")
                skipped += 1
        except Exception as e:
            logger.error(f"Error processing user: {e}")
            skipped += 1

    logger.info(f"Recommendations updated for {success} users. Skipped {skipped} users.")


if __name__ == "__main__":
    try:
        load_models()
        process_all_users()
    except Exception as e:
        logger.critical(f"Application failed: {e}")
