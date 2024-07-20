class SparseMatrix:
    """
    A sparse matrix.

    A standard 2D matrix takes up too much memory when storing the probabilities
    for the Markov chain. This utilizes far less memory by only retaining values
    explicitly set in the matrix.

    Usage::

        >>> mat = SparseMatrix()

        >>> mat.set("a", "a", 1)
        >>> mat.set("a", "b", 0.5)
        >>> mat.set("b", "c", 0.2)

        >>> mat.get("a", "b", default=0.9999)
        # 0.5

    """

    def __init__(self):
        self._names = []
        self._data = {}

    def __str__(self):
        return f"{len(self)} unique names"

    def __repr__(self):
        return f"<SparseSquareMatrix: {len(self)}>"

    def __len__(self):
        return len(self._names)

    def __contains__(self, key):
        return key in self._names

    def get(self, x, y, default=0.0):
        try:
            x_offset = self._names.index(x)
        except ValueError:
            return default

        row = self._data.get(y, {})
        return row.get(x_offset, default)

    def set(self, x, y, value):
        if x not in self._names:
            self._names.append(x)

        if y not in self._names:
            self._names.append(y)
            self._data[y] = {}

        x_offset = self._names.index(x)
        self._data.setdefault(y, {})
        row = self._data[y]
        row[x_offset] = value
