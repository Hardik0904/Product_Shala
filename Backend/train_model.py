"""
Trains the ProductShala sentiment LSTM and saves the model + tokenizer to
Backend/model/.

Data
----
Base data: UCI "Sentiment Labelled Sentences" dataset (Kotzias et al.,
KDD 2015) - 3000 short review sentences from Amazon, IMDB and Yelp, each
labelled 1 (positive) / 0 (negative). Source:
https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

That dataset barely contains any negation ("not good", "isn't great",
etc.), which is exactly the failure mode we're trying to fix. So we add a
small block of templated negation examples below to make sure the model
actually sees enough "not <positive word>" -> negative and
"not <negative word>" -> positive pairs to learn the pattern, instead of
just relying on organic examples the base dataset happens to contain.

Run:
    python train_model.py
"""

import os
import random

import dill as pk
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import preprocess_text

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CURR_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(CURR_DIR, "data", "raw")
MODEL_DIR = os.path.join(CURR_DIR, "model")

MAX_VOCAB = 8000
MAX_LEN = 60


def load_base_dataset() -> pd.DataFrame:
    frames = []
    for fname in ("amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"):
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- Templated negation examples --------------------------------------
# Small, deliberately repetitive templates so the model sees the SAME
# sentiment words appear both bare and negated, which is what teaches it
# that "not" flips the label rather than just being noise.
POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "fantastic", "wonderful",
    "impressive", "reliable", "comfortable", "satisfying", "helpful",
    "friendly", "fast", "worth it", "durable",
]
NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "disappointing", "poor", "useless",
    "annoying", "uncomfortable", "slow", "cheap", "unreliable",
    "frustrating", "broken", "horrible",
]
SUBJECTS = [
    "this product", "the service", "this phone", "the food", "this movie",
    "the app", "this laptop", "the staff", "the battery", "this item",
]
NEG_TEMPLATES = [
    "{subj} is not {word}",
    "{subj} was not {word}",
    "{subj} isn't {word}",
    "{subj} wasn't {word}",
    "honestly, {subj} is not {word} at all",
    "i wouldn't say {subj} is {word}",
]
PLAIN_TEMPLATES = [
    "{subj} is {word}",
    "{subj} was {word}",
    "{subj} is really {word}",
    "{subj} is so {word}",
]


def build_negation_augmentation() -> pd.DataFrame:
    rows = []
    for subj in SUBJECTS:
        for word in POSITIVE_WORDS:
            for tmpl in NEG_TEMPLATES:
                # "not good" -> negative sentiment (label 0)
                rows.append({"text": tmpl.format(subj=subj, word=word), "label": 0})
            for tmpl in PLAIN_TEMPLATES:
                # plain "good" -> positive (label 1), keeps the contrast pair
                rows.append({"text": tmpl.format(subj=subj, word=word), "label": 1})
        for word in NEGATIVE_WORDS:
            for tmpl in NEG_TEMPLATES:
                # "not bad" -> positive sentiment (label 1)
                rows.append({"text": tmpl.format(subj=subj, word=word), "label": 1})
            for tmpl in PLAIN_TEMPLATES:
                # plain "bad" -> negative (label 0)
                rows.append({"text": tmpl.format(subj=subj, word=word), "label": 0})
    df = pd.DataFrame(rows)
    # Downsample - we don't want the synthetic templates to totally swamp
    # the real, more linguistically varied review sentences.
    df = df.sample(n=min(2600, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    base_df = load_base_dataset()
    neg_df = build_negation_augmentation()
    df = pd.concat([base_df, neg_df], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

    print(f"Base dataset: {len(base_df)} rows")
    print(f"Negation-augmentation dataset: {len(neg_df)} rows")
    print(f"Total training rows: {len(df)}")

    df["clean_text"] = df["text"].astype(str).apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.15,
        random_state=SEED,
        stratify=df["label"],
    )

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LEN)
    x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LEN)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    model = tf.keras.Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LEN),
        Bidirectional(LSTM(48, return_sequences=False)),
        Dropout(0.4),
        Dense(24, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        x_train_seq,
        y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    loss, acc = model.evaluate(x_test_seq, y_test, verbose=0)
    print(f"\nHeld-out test accuracy: {acc:.4f} (loss {loss:.4f})")

    # Quick qualitative negation sanity-check
    print("\n--- Negation sanity check ---")
    sanity = [
        "This product is really good",
        "This product is not good",
        "This product is not bad",
        "The movie was terrible",
        "The movie wasn't terrible",
        "Great sound, but the battery is not reliable",
    ]
    sanity_clean = [preprocess_text(s) for s in sanity]
    sanity_seq = pad_sequences(tokenizer.texts_to_sequences(sanity_clean), maxlen=MAX_LEN)
    preds = model.predict(sanity_seq, verbose=0)
    for s, p in zip(sanity, preds):
        label = "Positive" if p[0] > 0.55 else ("Negative" if p[0] < 0.45 else "Neutral")
        print(f"{p[0]:.3f}  {label:<9} {s}")

    model.save(os.path.join(MODEL_DIR, "lstm_sentiment_model.keras"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pk.dump(tokenizer, f)

    print(f"\nSaved model + tokenizer to {MODEL_DIR}")


if __name__ == "__main__":
    main()
