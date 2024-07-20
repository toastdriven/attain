import copy
import csv
import json
import random

from . import matrix


class MarkovChain:
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
        return random.choice(self._matrix.names)

    def _get_state_offset(self, state):
        return self._matrix.x_offset(state)

    def create_transition_choices(self, transitions):
        choices = []
        magnitude = len(self)

        for offset, transition in enumerate(transitions):
            count_to_insert = int(transition * magnitude * 100)
            to_insert = offset
            choices.extend([to_insert for _ in range(count_to_insert)])

        random.shuffle(choices)
        return choices

    def generate(self, length=8, start_state=None):
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
        ending_punct = [".", "!", "?"]
        words = self.generate(length=random.randint(min_length, max_length + 1))
        sentence = " ".join(words) + random.choice(ending_punct)
        return sentence.capitalize()

    def to_csv(self, filename):
        with open(filename, "w") as raw_file:
            writer = csv.writer(raw_file)
            writer.writerow([""] + [state for state in self._matrix.names])

            for state in self._matrix.names:
                writer.writerow([state] + self._matrix.get_row(state))

    @classmethod
    def from_csv(cls, filename):
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
        with open(filename, "w") as raw_file:
            to_write = copy.deepcopy(self._matrix._data)
            to_write["__attain_headers__"] = copy.copy(self._matrix._names)
            json.dump(to_write, raw_file)

    @classmethod
    def from_json(cls, filename):
        mc = cls()

        with open(filename, "r") as raw_file:
            to_read = json.load(raw_file)
            mc._matrix._names = to_read.pop("__attain_headers__")

            for y_name, row_data in to_read.items():
                mc._matrix._data.setdefault(y_name, {})

                for key, values in row_data.items():
                    mc._matrix._data[y_name][int(key)] = values

        return mc
