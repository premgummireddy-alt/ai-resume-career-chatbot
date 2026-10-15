import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "app"))
from preprocess import normalize, sentence_chunks, dedupe

def test_normalize(): assert normalize("a  \n b") == "a b"
def test_chunks():
    c = sentence_chunks("One two. " * 100, max_words=20)
    assert all(len(x.split()) <= 20 for x in c)
def test_dedupe(): assert dedupe(["A", "a", "b"]) == ["A", "b"]
