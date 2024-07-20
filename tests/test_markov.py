import csv
from pathlib import Path
import random
import shutil

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
def data_dir():
    return Path("/tmp") / "attain_tests"


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


def test_train(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)
    assert len(mc) == 16


def test_dunder_methods(small_corpus):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    assert len(mc) == 16
    assert str(mc) == "16 known states"
    assert repr(mc) == "<MarkovChain: 16>"
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


def test_to_csv(small_corpus, data_dir):
    mc = markov.MarkovChain()
    mc.train(small_corpus)

    csv_path = data_dir / "transitions_matrix.csv"

    shutil.rmtree(data_dir.as_posix(), ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    assert csv_path.exists() == False

    mc.to_csv(csv_path.as_posix())

    assert csv_path.exists()

    with open(csv_path.as_posix(), "r") as raw_file:
        reader = csv.reader(raw_file)
        headers = next(reader)
        assert headers == [
            "",
            "world",
            "hello",
            "say",
            "to",
            "be",
            "or",
            "not",
            "that",
            "is",
            "the",
            "question",
            "kitty",
            "my",
            "favorite",
            "character",
            "here",
        ]

        row_0 = next(reader)
        assert row_0 == [
            "world",
            "0.0",
            "0.0",
            "0.045454545454545456",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
        ]
        row_1 = next(reader)
        assert row_1 == [
            "hello",
            "0.045454545454545456",
            "0.0",
            "0.0",
            "0.045454545454545456",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.045454545454545456",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
        ]

    shutil.rmtree(data_dir.as_posix(), ignore_errors=True)


def test_from_csv(small_corpus, data_dir):
    csv_path = data_dir / "transitions_matrix.csv"

    shutil.rmtree(data_dir.as_posix(), ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    assert csv_path.exists() == False

    with open(csv_path.as_posix(), "w") as raw_file:
        writer = csv.writer(raw_file)

        writer.writerow(["", "hello", "world", "peace"])
        writer.writerow(["hello", 0.25, 0.0, 0.0])
        writer.writerow(["world", 0.5, 0.0, 0.0])
        writer.writerow(["peace", 0.0, 0.25, 0.0])

    mc = markov.MarkovChain.from_csv(csv_path.as_posix())
    assert len(mc) == 3
    assert "hello" in mc
    assert "world" in mc
    assert "peace" in mc

    # Reach in a little, just to make sure things look right.
    assert mc._matrix.get_row("world") == [0.5, 0.0, 0.0]
    assert mc._matrix.get_column("hello") == [0.25, 0.5, 0.0]

    # To keep the randomness consistent.
    random.seed(a=1)

    assert mc.random_state() == "hello"

    shutil.rmtree(data_dir.as_posix(), ignore_errors=True)
