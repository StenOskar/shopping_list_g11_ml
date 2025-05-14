import tensorflow as tf
from tensorflow.keras import layers, Model


# This code defines a collaborative filtering model using TensorFlow and Keras.
class CollaborativeFilteringModel(Model):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        # Embeddings for users and items
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_dim,
            embeddings_initializer='he_normal'
        )
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_dim,
            embeddings_initializer='he_normal'
        )
        self.dot = layers.Dot(axes=1)

    # Define the model's input shape
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)

        # Reshape to ensure correct dimensions
        user_vec = tf.reshape(user_vec, [-1, user_vec.shape[-1]])
        item_vec = tf.reshape(item_vec, [-1, item_vec.shape[-1]])

        # Dot product
        score = self.dot([user_vec, item_vec])
        return score

    # Generate recommendations for a user
    def recommend_items(self, user_id, item_indices, top_k=5):
        # Create batch input
        user_indices = tf.fill([len(item_indices)], user_id)

        # Predict scores
        scores = self.predict([user_indices, item_indices], verbose=0).flatten()

        # Get indices of top items
        top_indices = tf.argsort(scores, direction='DESCENDING')[:top_k]
        return [item_indices[i] for i in top_indices.numpy()]