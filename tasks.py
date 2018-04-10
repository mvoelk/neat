
import numpy as np

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)

class Task(object):
    @property
    def  n_inputs(self):
        return np.shape(self.input_data)[1]
    @property
    def n_outputs(self):
        return np.shape(self.output_data)[1]

    def __init__(self):
        pass

    def evaluate(self):
        pass
        return [0, False]

    def test(self):
        pass

    def visualize(self):
        pass


# TODO
# tic-tac-toe
