import tensorflow as tf
from tensorflow.keras import layers, Model

# This is a simple collaborative filtering model using embeddings
class CollaborativeFilteringModel(Model):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        # Embeddings for users and items
        self.user_embedding = layers.Embedding(num_users, embedding_dim,
                                               embeddings_initializer='he_normal')
        self.item_embedding = layers.Embedding(num_items, embedding_dim,
                                               embeddings_initializer='he_normal')
        self.dot = layers.Dot(axes=1)

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


