#!/usr/bin/env python
import datetime
from pathlib import Path
import sys

import attain


def tokenize(sentence):
    tokens = []
    punct = "!@#$%^&*()_+-={}[]\\|;':\",.<>/?"

    for word in sentence.split(" "):
        cleaned = word
        cleaned = cleaned.strip()
        cleaned = cleaned.lower()
        cleaned = cleaned.rstrip(punct)
        cleaned = cleaned.lstrip(punct)

        if cleaned:
            tokens.append(cleaned)

    return tokens


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.datetime.now()

    def elapsed(self):
        return self.end - self.start


def main(corpus_filename):
    cache_filename = corpus_filename.parent.joinpath(f"{corpus_filename.stem}.json")
    mc = attain.MarkovChain()
    corpus = []

    if cache_filename.exists():
        mc = attain.MarkovChain.from_json(cache_filename.as_posix())
    else:
        with open(corpus_filename.as_posix(), "r") as raw_corpus:
            for line in raw_corpus:
                corpus.extend([word for word in tokenize(line)])

        with Timer() as t:
            mc.train(corpus)

        print(f"Trained in {t.elapsed()}")

        if not cache_filename.exists():
            mc.to_json(cache_filename.as_posix())

    return mc.generate_sentence()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_sentence.py <corpus_filename>")
        sys.exit(1)

    corpus_filename = Path(sys.argv[1])
    sentence = main(corpus_filename)
    print(sentence)
