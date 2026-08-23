"""
Shared text preprocessing for ProductShala sentiment model.

This module is imported by BOTH training (train_model.py) and inference
(data_predict.py) so the exact same cleaning + negation-marking logic is
applied at train and predict time. Keeping this in one place avoids the
classic bug where preprocessing silently drifts between training and
serving code.

Negation handling
------------------
A plain tokenizer treats "not good" as two independent tokens: "not" and
"good". A small LSTM trained on that will often still key off "good" and
predict positive, because it never learns that "not" flips the polarity
of the word(s) after it.

The standard fix (used in NLTK's `mark_negation` and many sentiment
pipelines) is to rewrite every word between a negation cue and the next
clause boundary (punctuation / conjunction) into its own "negated" token,
e.g.:

    "not good"          -> "not good_NEG"
    "not good, but..."  -> "not good_NEG but ..."   (stops at the comma)
    "isn't bad at all"  -> "is not bad_NEG at_NEG all_NEG"

This gives the model a distinct vocabulary item ("good_NEG") that it can
learn to associate with negative sentiment, separately from "good".
"""

import re

TAG_RE = re.compile(r"<[^>]+>")

# Contractions that carry a negation but don't contain the literal word
# "not" - expand these BEFORE negation marking so the cue word is visible.
CONTRACTIONS = {
    r"\bwon't\b": "will not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bain't\b": "is not",
    r"n't\b": " not",  # isn't/aren't/wasn't/doesn't/didn't/couldn't/... -> "is not"/"are not"/...
}

# Words that trigger "everything after this is negated" until a clause
# boundary is hit.
NEGATION_CUES = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "without", "hardly", "barely", "scarcely",
    "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt",
    "wont", "cant", "couldnt", "shouldnt", "wouldnt",
}

# Punctuation / conjunctions that end the scope of a negation.
CLAUSE_BOUNDARIES = {"but", "however", "though", "although", "yet"}

# These carry sentiment/intensity info and must survive stopword removal,
# even though a generic stopword list would normally strip them out.
CRITICAL_STOPWORDS = {
    "no", "not", "never", "none", "without", "should", "could", "might",
    "must", "will", "would", "very", "too", "only", "just", "even", "but",
}

GENERIC_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "of", "in", "on", "at", "to", "for", "with", "as", "by",
    "this", "that", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "their", "our", "your",
    "and", "or", "if", "then", "there", "here", "so", "such", "do",
    "does", "did", "has", "have", "had", "from", "into", "up", "down",
    "out", "about", "again", "further", "than", "s", "t", "can", "will",
}


def remove_tags(text: str) -> str:
    return TAG_RE.sub("", text)


def expand_contractions(text: str) -> str:
    for pattern, repl in CONTRACTIONS.items():
        text = re.sub(pattern, repl, text)
    return text


def mark_negation(tokens: list[str]) -> list[str]:
    """Suffix every token in the scope of a negation cue with '_NEG',
    stopping at punctuation-derived clause boundaries or conjunctions."""
    result = []
    negate = False
    for tok in tokens:
        if tok in CLAUSE_BOUNDARIES:
            negate = False
            result.append(tok)
            continue
        if tok in NEGATION_CUES:
            negate = True
            result.append(tok)
            continue
        if negate:
            result.append(f"{tok}_NEG")
        else:
            result.append(tok)
    return result


def preprocess_text(text: str) -> str:
    """Full cleaning pipeline: strip tags, lowercase, expand contractions,
    remove non-alpha chars (but keep clause-ending punctuation as a
    negation-scope boundary), mark negation scope, drop stopwords (while
    protecting the ones that carry sentiment/negation meaning)."""
    sentence = remove_tags(text).lower()
    sentence = expand_contractions(sentence)

    # Treat . , ! ? ; as clause boundaries by turning them into a marker
    # token BEFORE stripping punctuation, so mark_negation can see them.
    sentence = re.sub(r"[.,!?;]", " CLAUSE_END ", sentence)

    # Now strip anything that isn't a letter or our CLAUSE_END marker.
    sentence = re.sub(r"[^a-z\s_]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()

    tokens = sentence.split()
    # Swap CLAUSE_END markers into the boundary set mark_negation checks.
    tokens = ["but" if t == "CLAUSE_END" else t for t in tokens]

    tokens = mark_negation(tokens)

    cleaned = []
    for tok in tokens:
        base = tok[:-4] if tok.endswith("_NEG") else tok
        if len(base) <= 1:
            continue
        if base in GENERIC_STOPWORDS and base not in CRITICAL_STOPWORDS:
            continue
        cleaned.append(tok)

    return " ".join(cleaned)
