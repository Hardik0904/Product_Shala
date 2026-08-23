"""Lazily-loaded singletons for the Keras model and Keras tokenizer, so the
(fairly expensive) load-from-disk only happens once per process instead of
once per request."""

import os

import dill as pk
import tensorflow as tf

CURR_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURR_DIR, "model", "lstm_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(CURR_DIR, "model", "tokenizer.pkl")


class ModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if ModelLoader._instance is None:
            ModelLoader._instance = ModelLoader()
        return ModelLoader._instance

    def __init__(self) -> None:
        if ModelLoader._instance is not None:
            raise Exception("This is a singleton - use get_instance()")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python train_model.py` first to train and save it."
            )
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def get_data(self):
        return self.model


class TokenizerLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if TokenizerLoader._instance is None:
            TokenizerLoader._instance = TokenizerLoader()
        return TokenizerLoader._instance

    def __init__(self) -> None:
        if TokenizerLoader._instance is not None:
            raise Exception("This is a singleton - use get_instance()")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(
                f"Tokenizer file not found at {TOKENIZER_PATH}. "
                "Run `python train_model.py` first to train and save it."
            )
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pk.load(f)

    def get_data(self):
        return self.tokenizer
