""" examples.py

"""
import numpy
import numpy.random

import hmm.base

def hmm_simulate():
    """
>>> p_state_initial = [1, 0, 0]

>>> p_state_time_average = [0.5, 0.2, 0.3]

>>> p_state2state = [
...    [0,   0.5, 0.5],
...    [0.5, 0.5, 0  ],
...    [0.5, 0,   0.5]
...    ]

>>> model_py_state = numpy.array([
...    [0.5, 0.5],
...    [0.8, 0.2],
...    [0.2, 0.8],
...    ])

>>> random_number_generator = numpy.random.default_rng(0)
>>> y_model = hmm.base.Observation(model_py_state, random_number_generator)

>>> hmm = hmm.base.HMM(p_state_initial, p_state_time_average, p_state2state,
...  y_model, random_number_generator)

>>> states, ys = hmm.simulate(10)
>>> print(states)
[1, 0, 2, 2, 2, 2, 2, 2, 2, 0]
>>> print(ys)
[0, 0, 1, 1, 1, 0, 0, 0, 1, 0]

    """
    pass

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
#--------------------------------
# Local Variables:
# mode: python
# End:
