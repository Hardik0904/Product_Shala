from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_loader import ModelLoader, TokenizerLoader
from preprocessing import preprocess_text

MAX_LEN = 60  # must match MAX_LEN used in train_model.py

TOKENIZER_LOADER = TokenizerLoader.get_instance()
lstm_tokenizer = TOKENIZER_LOADER.get_data()

MODEL_LOADER = ModelLoader.get_instance()
lstm_model = MODEL_LOADER.get_data()


def model_predict(text: str, tokenizer=lstm_tokenizer, model=lstm_model):
    """Returns (label: str, polarity: float) where polarity is the raw
    sigmoid output in [0, 1] - closer to 1 is more positive."""
    clean_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    polarity = float(model.predict(padded_sequence, verbose=0)[0][0])

    if polarity < 0.45:
        label = "Negative"
    elif polarity > 0.55:
        label = "Positive"
    else:
        label = "Neutral"

    return label, polarity


if __name__ == "__main__":
    samples = [
        "This product is really good",
        "This product is not good",
        "This product is not bad",
        "Absolutely terrible, would not buy again",
        "It's okay I guess, nothing special",
    ]
    for s in samples:
        label, polarity = model_predict(s)
        print(f"{polarity:.3f}  {label:<9} {s}")
