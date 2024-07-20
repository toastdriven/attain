import copy
import csv
import json
import random

from . import matrix


class MarkovChain:
    """
    An object for training/generating Markov Chains.

    Based on a given corpus, the Markov Chain generates a sequence based on
    how likely two "things" are to appear together.

    Usage::

        >>> mc = MarkovChain()

        >>> mc.train(["big", "seq", "of", "prepped", "words"])

        >>> mc.generate(length=4)
        ["big", "words", "of", "seq"]

        >>> mc.generate_sentence()
        "Of words of big seq?"

    """

    def __init__(self):
        self._matrix = matrix.SparseMatrix()

    def __str__(self):
        return f"{len(self)} known states"

    def __repr__(self):
        return f"<MarkovChain: {len(self)}>"

    def __contains__(self, value):
        return value in self._matrix

    def __len__(self):
        return len(self._matrix)

    def train(self, seq):
        """
        Trains the Markov Chain against a corpus.

        Can only be called once against the full dataset. If your dataset
        grows/changes, you need to re-train.

        Args:
            seq (iterable): A training list of (prepared) words/states from the Real
                World(tm).
        """
        last_state = None
        total_count = len(seq)
        incr_by = 1 / (total_count - 1)

        for current_state in seq:
            # Special case for the first state. Nothing to update.
            if last_state is None:
                last_state = current_state
                continue

            # Increment the correct transition.
            current_value = self._matrix.get(current_state, last_state)
            self._matrix.set(current_state, last_state, current_value + incr_by)

            # Finally, swap the current into the last state.
            last_state = current_state

    def random_state(self):
        """
        Selects a random state from the trained options.

        Returns:
            str: The name of the random state.
        """
        return random.choice(self._matrix.names)

    def _get_state_offset(self, state):
        return self._matrix.x_offset(state)

    def create_transition_choices(self, transitions):
        """
        Given a set of known transitions, creates a list of randomized choices
        with representative likelihoods.

        Args:
            transitions (list): A sparse list of transitions/percentages.

        Returns:
            list: A large, representative list of all the state (offsets) in
                randomized order.
        """
        choices = []
        magnitude = len(self)

        for offset, transition in enumerate(transitions):
            count_to_insert = int(transition * magnitude * 100)
            to_insert = offset
            choices.extend([to_insert for _ in range(count_to_insert)])

        random.shuffle(choices)
        return choices

    def generate(self, length=8, start_state=None):
        """
        Selects a series of choices based on probability/chance, with each
        building on the previous selection.

        Args:
            length (int): The number of choices to generate. Default is `8`.
            start_state (str): [Optional] The first choice to begin the selections
                with.

        Returns:
            list: The generated Markov chain.
        """
        states = []
        last_state = None

        if start_state is None:
            states.append(self.random_state())
            last_state = states[0]

        for _ in range(length - 1):
            choices = self.create_transition_choices(self._matrix.get_row(last_state))

            while len(choices) <= 0:
                choices = self.create_transition_choices(
                    self._matrix.get_row(self.random_state())
                )

            choice = self._matrix.names[random.choice(choices)]
            states.append(choice)
            last_state = choice

        return states

    def generate_sentence(self, min_length=5, max_length=10):
        """
        Attempts to create an English-like sentence from the generated Markov Chain.

        Will generate a chain of random length (between the provided `min_length`
        & `max_length`), as well as capitalizing the sentence, & adding random
        punctuation.

        Args:
            min_length (int): [Optional] The fewest number of choices in the chain.
                Default is `5`.
            max_length (int): [Optional] The most number of choices in the chain.
                Default is `10`.

        Returns:
            str: A vaguely English-like collection of words emulating a sentence.
        """
        ending_punct = [".", "!", "?"]
        words = self.generate(length=random.randint(min_length, max_length + 1))
        sentence = " ".join(words) + random.choice(ending_punct)
        return sentence.capitalize()

    def to_csv(self, filename):
        """
        Exports the trained model to a (verbose) CSV file.

        Warning: On big models/training sets, this can get very large.

        Args:
            filename (str): The filename to write the CSV data to.
        """
        with open(filename, "w") as raw_file:
            writer = csv.writer(raw_file)
            writer.writerow([""] + [state for state in self._matrix.names])

            for state in self._matrix.names:
                writer.writerow([state] + self._matrix.get_row(state))

    @classmethod
    def from_csv(cls, filename):
        """
        Imports the trained model from a (verbose) CSV file.

        Warning: On big models/training sets, this can get very large.

        Args:
            filename (str): The filename to read the CSV data from.
        """
        mc = cls()

        with open(filename, "r") as raw_file:
            reader = csv.reader(raw_file)
            headers = next(reader)
            # Skip the blank space in the top-left.
            mc._matrix._names = headers[1:]

            for row in reader:
                y_name = row[0]

                for offset, value in enumerate(row[1:]):
                    x_name = mc._matrix._names[offset]
                    value = float(value)

                    if value != mc._matrix._default:
                        mc._matrix.set(x_name, y_name, value)

        return mc

    def to_json(self, filename):
        """
        Exports the trained model to a (sparse) JSON file.

        Args:
            filename (str): The filename to write the JSON data to.
        """
        with open(filename, "w") as raw_file:
            to_write = copy.deepcopy(self._matrix._data)
            to_write["__attain_headers__"] = copy.copy(self._matrix._names)
            json.dump(to_write, raw_file)

    @classmethod
    def from_json(cls, filename):
        """
        Imports the trained model from a (sparse) JSON file.

        Args:
            filename (str): The filename to read the JSON data from.
        """
        mc = cls()

        with open(filename, "r") as raw_file:
            to_read = json.load(raw_file)
            mc._matrix._names = to_read.pop("__attain_headers__")

            for y_name, row_data in to_read.items():
                mc._matrix._data.setdefault(y_name, {})

                for key, values in row_data.items():
                    mc._matrix._data[y_name][int(key)] = values

        return mc
