from functools import cached_property


class SparseMatrix:
    """
    A sparse matrix.

    A standard 2D matrix takes up too much memory when storing the probabilities
    for the Markov chain. This utilizes far less memory by only retaining values
    explicitly set in the matrix.

    Usage::

        # The default here is optional (typically `0.0`).
        >>> mat = SparseMatrix(default=0.9999)

        >>> mat.set("a", "a", 1)
        >>> mat.set("a", "b", 0.5)
        >>> mat.set("b", "c", 0.2)

        >>> mat.get("a", "b")
        # 0.5

    """

    def __init__(self, default=0.0):
        self._x_labels = {}
        self._y_labels = {}
        self._x_inverted_cache = None
        self._y_inverted_cache = None
        # The `_data` is a dict-of-dicts; in row-column-value order.
        self._data = {}
        self._default = default

    def __str__(self):
        return f"{len(self._x_labels)}x{len(self._y_labels)}"

    def __repr__(self):
        return f"<SparseMatrix: {len(self._x_labels)}x{len(self._y_labels)}>"

    def __len__(self):
        return len(self._x_labels)

    def __contains__(self, key):
        return key in self._x_labels

    @property
    def x_labels(self):
        return self._x_labels.keys()

    @property
    def y_labels(self):
        return self._y_labels.keys()

    @cached_property
    def x_labels_in_order(self):
        labels = sorted(
            [[key, value] for key, value in self._x_labels.items()],
            key=lambda x: x[1],
        )
        return [label[0] for label in labels]

    @cached_property
    def y_labels_in_order(self):
        labels = sorted(
            [[key, value] for key, value in self._y_labels.items()],
            key=lambda x: x[1],
        )
        return [label[0] for label in labels]

    def x_offset(self, x):
        """
        Returns the numerical offset of label `x`.

        Args:
            x (str): The column name.

        Returns:
            int: The column offset. If not found, returns `None`.
        """
        return self._x_labels.get(x)

    def y_offset(self, y):
        """
        Returns the numerical offset of label `y`.

        Args:
            y (str): The row name.

        Returns:
            int: The row offset. If not found, returns `None`.
        """
        return self._y_labels.get(y)

    def label_at_x_offset(self, x_offset):
        """
        Returns the label at numerical offset `x_offset`.

        Args:
            x_offset (int): The column number.

        Returns:
            str: The column label. If not found, returns `None`.
        """
        if self._x_inverted_cache is None:
            self._x_inverted_cache = {v: k for k, v in self._x_labels.items()}

        return self._x_inverted_cache.get(x_offset)

    def label_at_y_offset(self, y_offset):
        """
        Returns the label at numerical offset `y_offset`.

        Args:
            y_offset (int): The row number.

        Returns:
            str: The row label. If not found, returns `None`.
        """
        if self._y_inverted_cache is None:
            self._y_inverted_cache = {v: k for k, v in self._y_labels.items()}

        return self._y_inverted_cache.get(y_offset)

    def get(self, x, y):
        """
        Fetches the value at a given column/row.

        If not set, returns the default value.

        Args:
            x (str): The column name.
            y (str): The row name.

        Returns:
            any: The value at that location. If not set, returns the default
                value for the `SparseMatrix`.
        """
        x_offset = self.x_offset(x)
        y_offset = self.y_offset(y)

        if x_offset is None or y_offset is None:
            return self._default

        row = self._data.get(y_offset, {})
        return row.get(x_offset, self._default)

    def set(self, x, y, value):
        """
        Sets a value at a given column/row location.

        Args:
            x (str): The column name.
            y (str): The row name.
            value (any): The value to place at that location.

        Returns:
            None
        """
        # Clear the caches.
        self._x_inverted_cache = None
        self._y_inverted_cache = None

        x_offset = self.x_offset(x)
        y_offset = self.y_offset(y)

        if x_offset is None:
            x_offset = len(self._x_labels)
            self._x_labels[x] = x_offset

        if y_offset is None:
            y_offset = len(self._y_labels)
            self._y_labels[y] = y_offset

        self._data.setdefault(y_offset, {})
        self._data[y_offset].setdefault(x_offset, {})
        self._data[y_offset][x_offset] = value

    def get_sparse_row(self, y):
        """
        Returns a sparse "row" (dict) for a given row.

        Mostly internal, but there if you need it.

        Args:
            y (str): The row name.

        Returns:
            dict: A sparse representation of the row. Keys are **offsets** of the
                columns, values are the value at locations that are set.
        """
        row_data = {}
        y_offset = self.y_offset(y)

        for x_offset, data in self._data.get(y_offset, {}).items():
            row_data[self.label_at_x_offset(x_offset)] = data

        return row_data

    def get_sparse_column(self, x):
        """
        Returns a sparse "column" (dict) for a given column.

        Mostly internal, but there if you need it.

        Args:
            x (str): The column name.

        Returns:
            dict: A sparse representation of the column. Keys are **offsets** of
                the columns, values are the value at locations that are set.
        """
        column_data = {}
        x_offset = self.x_offset(x)

        for y_offset, data in self._data.items():
            if x_offset in data:
                column_data[self.label_at_y_offset(y_offset)] = data[x_offset]

        return column_data

    def get_row(self, y):
        """
        Returns a full row, as if the matrix were fully populated.

        Args:
            y (str): The row name.

        Returns:
            list: A full representation of the row, populated with default values
                where not explicitly set.
        """
        row_data = []
        sparse_row = self.get_sparse_row(y)
        x_offsets_in_order = sorted(self._x_inverted_cache)

        for offset in x_offsets_in_order:
            x_label = self._x_inverted_cache[offset]
            row_data.append(sparse_row.get(x_label, self._default))

        return row_data

    def get_column(self, x):
        """
        Returns a full column, as if the matrix were fully populated.

        Args:
            x (str): The column name.

        Returns:
            list: A full representation of the column, populated with default values
                where not explicitly set.
        """
        column_data = []
        sparse_col = self.get_sparse_column(x)
        y_offsets_in_order = sorted(self._y_inverted_cache)

        for offset in y_offsets_in_order:
            y_label = self._y_inverted_cache[offset]
            column_data.append(sparse_col.get(y_label, self._default))

        return column_data
