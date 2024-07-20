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
        self._names = []
        self._data = {}
        self._default = default

    def __str__(self):
        return f"{len(self)} unique names"

    def __repr__(self):
        return f"<SparseMatrix: {len(self)}>"

    def __len__(self):
        return len(self._names)

    def __contains__(self, key):
        return key in self._names

    @property
    def names(self):
        return self._names

    def x_offset(self, x):
        """
        Returns the numerical offset of name `x`.

        Mostly internally useful for determining column names within a row.

        Args:
            x (str): The column name.

        Returns:
            int: The column offset.
        """
        return self._names.index(x)

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
        try:
            x_offset = self._names.index(x)
        except ValueError:
            return self._default

        row = self._data.get(y, {})
        return row.get(x_offset, self._default)

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
        return self._data.get(y, {})

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
        column = {}
        column_offset = self.x_offset(x)

        for offset, x_name in enumerate(self._names):
            row = self._data.get(x_name, {})

            if column_offset in row:
                column[offset] = row[column_offset]

        return column

    def get_row(self, y):
        """
        Returns a full row, as if the matrix were fully populated.

        Args:
            y (str): The row name.

        Returns:
            list: A full representation of the row, populated with default values
                where not explicitly set.
        """
        actual_row = self.get_sparse_row(y)
        row = []

        for offset, _ in enumerate(self._names):
            row.append(actual_row.get(offset, self._default))

        return row

    def get_column(self, x):
        """
        Returns a full column, as if the matrix were fully populated.

        Args:
            x (str): The column name.

        Returns:
            list: A full representation of the column, populated with default values
                where not explicitly set.
        """
        column = []
        offset = self.x_offset(x)

        # This is a little slow & kinda-garbage.
        for x_name in self._names:
            row = self._data.get(x_name, {})
            column.append(row.get(offset, self._default))

        return column

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
        if x not in self._names:
            self._names.append(x)

        if y not in self._names:
            self._names.append(y)
            self._data[y] = {}

        x_offset = self._names.index(x)
        self._data.setdefault(y, {})
        row = self._data[y]
        row[x_offset] = value
