import tensorflow as tf
import numpy as np


class RecommendationModel:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_users, 50),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(self.num_items, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, user_data, item_data):
        self.model.fit(user_data, item_data, epochs=5)

    def save_model(self, path="models/recommendation_model.h5"):
        self.model.save(path)

    def load_model(self, path="models/recommendation_model.h5"):
        self.model = tf.keras.models.load_model(path)