

import bt
import numpy

# ---------------------------------------------------------------

class RunRandomly(bt.algos.Algo):

    """
    Returns True on first run then returns False.

    Args:
        * run_on_first_call: bool which determines if it runs the first time the algo is called

    As the name says, the algo only runs once. Useful in situations
    where we want to run the logic once (buy and hold for example).

    """

    prob: float

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def __call__(self, target):
        return numpy.random.choice(
            [True, False], p = [self.prob, 1 - self.prob]
        )

# ---------------------------------------------------------------
