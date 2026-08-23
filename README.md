# ProductShala (rebuilt)

A full-stack app that classifies product/service reviews as Positive,
Negative, or Neutral using a Bidirectional LSTM, with explicit handling
for negation ("not good", "isn't great", "not bad") that the original
version couldn't do. Backend is Flask, frontend is React (Vite) — no
frameworks or pages left over from the old repo that referenced files
that didn't actually exist.

## Quick start

```bash
# Terminal 1 - backend
cd Backend
pip install -r requirements.txt
python app.py            # -> http://localhost:5000

# Terminal 2 - frontend
cd Frontend
npm install
npm run dev               # -> http://localhost:5173
```

Open the frontend URL, type a review (or click one of the example chips),
and hit "Stamp it".

## What changed from the original version

**Backend**
1. **The trained model was never actually in the repo.** The old
   `.gitignore` excluded `*.keras` / `*.pkl`, so `Backend/model/` didn't
   exist after cloning and the app crashed on startup. This version
   commits the (small, ~4MB) model files directly.
2. **Negation handling.** `preprocessing.py` rewrites words in the scope
   of a negation cue into distinct tokens, e.g. `"not good"` →
   `not good_NEG`. This gives the model a separate vocabulary item for
   "good-negated" vs "good", instead of relying on it to somehow infer
   that "not" flips meaning. The training data is also augmented with
   ~2,600 templated negation examples (`"X is not good"` vs `"X is
   good"`, etc.) since the real dataset barely contains any negation
   examples on its own.
3. **Same preprocessing at train and predict time.** Both
   `train_model.py` and `data_predict.py` import the *same*
   `preprocess_text()` function from `preprocessing.py`, so there's no
   risk of the cleaning logic drifting between training and serving
   (a common source of silent bugs).
4. Removed the leftover hardcoded `/data` test endpoint from `app.py`.

**Frontend**
1. **`react-router-dom` was imported but never listed as a dependency** —
   `npm install` followed by `npm run dev` would crash immediately.
2. **`SignupPage` and `DashBoard` were imported in `App.jsx` but the
   files didn't exist in the repo** — another guaranteed crash on load.
3. **The API call was hardcoded to `http://172.20.10.3:5000`** — someone's
   personal WiFi IP, not `localhost`, so it could never have worked for
   anyone else who cloned the repo.
4. Rebuilt as a single focused page (the old Login/Signup/About/Contact/
   Dashboard pages were unused stubs, not part of the actual product) —
   a "review slip" you fill in that gets stamped with the sentiment
   verdict, live against the real model.

## Project structure

```
Backend/
 ├── app.py              # Flask API (/predict)
 ├── preprocessing.py    # shared text cleaning + negation marking
 ├── train_model.py       # trains and saves the LSTM + tokenizer
 ├── data_loader.py       # loads model/tokenizer as singletons
 ├── data_predict.py      # inference wrapper used by app.py
 ├── requirements.txt
 ├── data/raw/            # UCI Sentiment Labelled Sentences dataset (txt)
 └── model/               # trained model + tokenizer (committed)

Frontend/
 ├── index.html
 ├── package.json
 ├── .env.example         # VITE_API_URL, copy to .env to override
 └── src/
     ├── App.jsx
     ├── api.js           # fetch wrapper for the Flask backend
     └── components/
         ├── ReviewSlip.jsx / .css   # the input + submit "hero"
         ├── StampBadge.jsx / .css   # the ink-stamp verdict visual
         └── HowItWorks.jsx / .css   # 3-step pipeline explainer
```

## Backend setup

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

The API starts on `http://localhost:5000`.

## Frontend setup

```bash
cd Frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` by default and talks to
`http://localhost:5000` (see `.env.example` — copy to `.env` and set
`VITE_API_URL` if your backend runs elsewhere).

### Retraining the model

```bash
cd Backend
python train_model.py
```

This regenerates `model/lstm_sentiment_model.keras` and
`model/tokenizer.pkl`. Do this if you change `preprocessing.py` or want
to add more training data — the model and preprocessing logic must
always be kept in sync.

## API

**POST `/predict`**

Accepts either form data or JSON:

```bash
curl -X POST http://localhost:5000/predict -F "review=This product is not bad"
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "The battery life is terrible"}'
```

Response:

```json
{
  "review": "This product is not bad",
  "prediction": "Positive",
  "polarity": 0.956
}
```

`polarity` is the raw model output in `[0, 1]`; closer to 1 is more
positive. Labels: `< 0.45` → Negative, `> 0.55` → Positive, else Neutral.

## Data

Base training data is the **UCI Sentiment Labelled Sentences** dataset
(Kotzias et al., KDD 2015) — 3,000 short review sentences from Amazon,
IMDB, and Yelp, each labelled positive/negative. On top of that,
`train_model.py` generates templated negation examples to specifically
teach the model that "not <positive word>" is negative and "not
<negative word>" is positive.

Held-out test accuracy: **~89%**. Sanity check on negation cases (see
`train_model.py` output):

```
1.000  Positive  This product is really good
0.004  Negative  This product is not good
0.956  Positive  This product is not bad
0.010  Negative  The movie was terrible
0.956  Positive  The movie wasn't terrible
```

## Known limitations

- **Sarcasm is not handled.** ("Oh great, another broken phone" will
  likely be read as positive because of "great".) This needs real
  contextual/world knowledge that a small LSTM over a few thousand
  sentences can't learn — realistically this needs a fine-tuned
  transformer (e.g. DistilBERT) trained on sarcasm-labelled data
  (SARC, iSarcasm) to meaningfully improve.
- The base dataset is small (3,000 real sentences) and skews toward
  short, simple sentences from Amazon/IMDB/Yelp circa 2015 — it won't
  generalize perfectly to every review style or domain.
- Double negatives / more complex negation scope (e.g. negation
  spanning multiple clauses) are only partially covered by the
  templated augmentation.
