"""test_observe_float.py: T run "$ python -m pytest test_observe_float.py" or
"$ python -m pytest hmm/tests"

"""
# Copyright (c) 2021 Andrew M. Fraser
import unittest

import numpy
import numpy.testing

import scipy.linalg

import hmm.observe_float
import hmm.base


class TestGauss(unittest.TestCase):
    """ Test hmm.observe_float.Gauss
    """

    def setUp(self):

        p_initial_state = [0.67, 0.33]
        p_state2state = [[0.93, 0.07], [0.13, 0.87]]
        mu_1 = numpy.array([-1.0, 1.0])
        var_1 = numpy.ones(2)
        self.rng = numpy.random.default_rng(0)
        y_mod = hmm.observe_float.Gauss(mu_1.copy(), var_1.copy(), self.rng)
        self.model_1_1 = hmm.base.HMM(p_initial_state, p_initial_state,
                                      p_state2state, y_mod, self.rng)
        self.model_2_4 = hmm.base.HMM(
            p_initial_state, p_initial_state, p_state2state,
            hmm.observe_float.Gauss(mu_1 * 2, var_1 * 4, self.rng))

        # Exercises random_out
        _, y_train = self.model_1_1.simulate(100)

        self.y_train = numpy.array(y_train, numpy.float64).reshape((-1,))

    def test_decode(self):
        rv = numpy.array(self.model_1_1.decode((self.y_train,)))
        self.assertTrue(rv.sum() == 49)

    def test_train(self):
        self.model_2_4.y_mod.observe((self.y_train,))

        # Exercises calculate and reestimate
        rv = numpy.array(self.model_2_4.train((self.y_train,), n_iterations=15))

        difference = rv[1:] - rv[:-1]
        self.assertTrue(difference.min() > 0)  # Check monotonic

    def test_str(self):
        string = self.model_1_1.y_mod.__str__()
        n_instance = string.find('instance')
        tail = string[n_instance:]
        self.assertTrue(
            tail == 'instance:\n    mu\n[-1.  1.]\n    var\n[1. 1.]\n')


# --------------------------------
# Local Variables:
# mode: python
# End:
