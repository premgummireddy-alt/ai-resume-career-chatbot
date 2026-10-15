"""NLTK-based dataset cleaning and chunking."""
import re
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

def normalize(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def sentence_chunks(text: str, max_words: int = 120):
    try:
        sents = nltk.sent_tokenize(normalize(text))
    except LookupError:
        sents = re.split(r"(?<=[.!?])\s+", normalize(text))
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur)); cur = []
        cur.append(s)
    if cur: chunks.append(" ".join(cur))
    return chunks

def dedupe(chunks):
    seen, out = set(), []
    for c in chunks:
        k = c.lower()
        if k not in seen:
            seen.add(k); out.append(c)
    return out
