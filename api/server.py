from fastapi import FastAPI
from models.recommender import RecommendationModel


# This class will load the model and serves recommendations via FastAPI
app = FastAPI()

