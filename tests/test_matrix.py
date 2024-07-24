import pytest

from attain import matrix


@pytest.fixture
def small_matrix():
    """
    Matrix:
    |       | hello | world |
    | ----- | ----- | ----- |
    | hello |       |       |
    | world | 0.5   |       |

    Internal:
    _x_labels = {"hello": 0}
    _y_labels = {"world": 0}
    {
        0: {
            0: 0.5
        }
    }
    """
    mat = matrix.SparseMatrix()
    mat.set("hello", "world", 0.5)
    return mat


@pytest.fixture
def medium_matrix():
    """
    Matrix:
    |   | a   | b   | c   | d   |
    | - | --- | --- | --- | --- |
    | a | 1   |     | 0.3 |     |
    | b | 0.5 | 0.2 |     |     |
    | d |     |     | 1   |     |
    | c |     |     |     | 0.6 |

    Internal:
    _x_labels = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
    }
    _y_labels = {
        "a": 0,
        "b": 1,
        "c": 3,
        "d": 2,
    }
    {
        # Y-key
        0: {
            # X-key: Value
            0: 1,
            2: 0.3,
        },
        1: {
            0: 0.5,
            1: 0.2,
        },
        2: {
            2: 1,
        },
        3: {
            3: 0.6,
        },
    }
    """
    mat = matrix.SparseMatrix()
    mat.set("a", "a", 1)
    mat.set("a", "b", 0.5)
    mat.set("b", "b", 0.2)
    mat.set("c", "a", 0.3)
    mat.set("c", "d", 1)
    mat.set("d", "c", 0.6)
    return mat


def test_matrix_set():
    mat = matrix.SparseMatrix()
    mat.set("hello", "world", 0.5)

    assert mat._x_labels == {
        "hello": 0,
    }
    assert mat._y_labels == {
        "world": 0,
    }
    assert mat._data == {
        0: {
            0: 0.5,
        },
    }


def test_small_matrix_internals(small_matrix):
    assert small_matrix._x_labels == {
        "hello": 0,
    }
    assert small_matrix._y_labels == {
        "world": 0,
    }
    assert small_matrix._data == {
        0: {
            0: 0.5,
        },
    }


def test_small_matrix_get(small_matrix):
    assert small_matrix.get("hello", "world") == 0.5

    # Not found, return default.
    assert small_matrix.get("hello", "hello") == 0.0


def test_medium_matrix_internals(medium_matrix):
    assert medium_matrix._x_labels == {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
    }
    assert medium_matrix._y_labels == {
        "a": 0,
        "b": 1,
        "c": 3,
        "d": 2,
    }
    assert medium_matrix._data == {
        0: {
            0: 1,
            2: 0.3,
        },
        1: {
            0: 0.5,
            1: 0.2,
        },
        2: {
            2: 1,
        },
        3: {
            3: 0.6,
        },
    }


def test_medium_matrix_get(medium_matrix):
    assert medium_matrix.get("a", "a") == 1
    assert medium_matrix.get("a", "b") == 0.5
    assert medium_matrix.get("b", "b") == 0.2
    assert medium_matrix.get("c", "a") == 0.3
    assert medium_matrix.get("c", "d") == 1
    assert medium_matrix.get("d", "c") == 0.6

    # Not found, return default.
    assert medium_matrix.get("q", "r") == 0.0


def test_x_offset(medium_matrix):
    assert medium_matrix.x_offset("a") == 0
    assert medium_matrix.x_offset("b") == 1
    assert medium_matrix.x_offset("c") == 2
    assert medium_matrix.x_offset("d") == 3


def test_y_offset(medium_matrix):
    assert medium_matrix.y_offset("a") == 0
    assert medium_matrix.y_offset("b") == 1
    assert medium_matrix.y_offset("d") == 2
    assert medium_matrix.y_offset("c") == 3


def test_get_sparse_row(medium_matrix):
    assert medium_matrix.get_sparse_row("a") == {"a": 1, "c": 0.3}
    assert medium_matrix.get_sparse_row("b") == {"a": 0.5, "b": 0.2}
    assert medium_matrix.get_sparse_row("c") == {"d": 0.6}
    assert medium_matrix.get_sparse_row("d") == {"c": 1}


def test_get_sparse_column(medium_matrix):
    assert medium_matrix.get_sparse_column("a") == {"a": 1, "b": 0.5}
    assert medium_matrix.get_sparse_column("b") == {"b": 0.2}
    assert medium_matrix.get_sparse_column("c") == {"a": 0.3, "d": 1}
    assert medium_matrix.get_sparse_column("d") == {"c": 0.6}


def test_get_row(medium_matrix):
    assert medium_matrix.get_row("a") == [1, 0.0, 0.3, 0.0]
    assert medium_matrix.get_row("b") == [0.5, 0.2, 0.0, 0.0]
    assert medium_matrix.get_row("c") == [0.0, 0.0, 0.0, 0.6]
    assert medium_matrix.get_row("d") == [0.0, 0.0, 1, 0.0]


def test_get_column(medium_matrix):
    assert medium_matrix.get_column("a") == [1, 0.5, 0.0, 0.0]
    assert medium_matrix.get_column("b") == [0.0, 0.2, 0.0, 0.0]
    assert medium_matrix.get_column("c") == [0.3, 0.0, 1, 0.0]
    assert medium_matrix.get_column("d") == [0.0, 0.0, 0.0, 0.6]


def test_str(medium_matrix):
    assert str(medium_matrix) == "4x4"


def test_repr(medium_matrix):
    assert repr(medium_matrix) == "<SparseMatrix: 4x4>"


def test_len(medium_matrix):
    assert len(medium_matrix) == 4


def test_contains(medium_matrix):
    assert "a" in medium_matrix
    assert "q" not in medium_matrix


def test_x_labels(medium_matrix):
    assert medium_matrix.x_labels_in_order == ["a", "b", "c", "d"]


def test_y_labels(medium_matrix):
    assert medium_matrix.y_labels_in_order == ["a", "b", "d", "c"]
