import random

import pytest

from attain import markov


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


@pytest.fixture
def small_corpus():
    sentences = [
        "Hello, world!",
        "Say hello.",
        "To be or not to be, that is the question.",
        "Hello Kitty is my favorite character.",
        "Here, kitty kitty.",
    ]
    corpus = []

    for sentence in sentences:
        corpus.extend(tokenize(sentence))

    return corpus


def test_initial():
    mc = markov.MarkovChain()
    assert len(mc) == 0


def test_add():
    mc = markov.MarkovChain()
    assert len(mc) == 0

    mc.add_state("hello")
    assert len(mc) == 1
    assert "hello" in mc


def test_train(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)
    assert len(mc) == 16


def test_dunder_methods(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    assert len(mc) == 16
    assert str(mc) == "16 known states"
    assert repr(mc) == "<MarkovChain: 16 known states>"
    assert "hello" in mc
    assert "whatever" not in mc


def test_random_state(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    # To keep the randomness consistent.
    random.seed(a=1)

    assert mc.random_state() == "be"
    assert mc.random_state() == "say"
    assert mc.random_state() == "is"


def test_generate(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    # To keep the randomness consistent.
    random.seed(a=1)

    assert mc.generate() == [
        "be",
        "that",
        "is",
        "the",
        "question",
        "hello",
        "to",
        "be",
    ]
    assert mc.generate(length=4) == [
        "my",
        "favorite",
        "character",
        "here",
    ]


def test_generate_sentence(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    # To keep the randomness consistent.
    random.seed(a=1)

    assert mc.generate_sentence() == "Say hello kitty is my favorite."
    assert mc.generate_sentence() == "Kitty is the question hello kitty is my favorite!"
    assert mc.generate_sentence(min_length=3, max_length=4) == "Not to be that?"
