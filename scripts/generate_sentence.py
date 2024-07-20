#!/usr/bin/env python
import sys

import attain


def tokenize(sentence):
    tokens = []
    punct = "!@#$%^&*()_+-={}[]\|;':\",.<>\/?"

    for word in sentence.split(" "):
        cleaned = word
        cleaned = cleaned.strip()
        cleaned = cleaned.lower()
        cleaned = cleaned.rstrip(punct)
        cleaned = cleaned.lstrip(punct)

        if cleaned:
            tokens.append(cleaned)

    return tokens


def main(corpus_filename):
    mc = attain.MarkovChain()
    corpus = []

    with open(corpus_filename, "r") as raw_corpus:
        for line in raw_corpus:
            corpus.extend([word for word in tokenize(line)])

    mc.train(corpus)
    return mc.generate_sentence()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_sentence.py <corpus_filename>")
        sys.exit(1)

    sentence = main(sys.argv[1])
    print(sentence)
