"""test_observe_float.py: T run "$ python -m pytest test_observe_float.py" or
"$ python -m pytest hmm/tests"

"""
# Copyright (c) 2021 Andrew M. Fraser
import unittest

import numpy as np
import numpy.testing

import scipy.linalg

import hmm.observe_float
import hmm.extensions


class TestGauss(unittest.TestCase):
    """ Test hmm.observe_float.Gauss TODO: improve these tests.
    """

    def setUp(self):

        p_s0 = [0.67, 0.33]
        p_ss = [[0.93, 0.07], [0.13, 0.87]]
        mu_1 = np.array([-1.0, 1.0])
        var_1 = np.ones(2)
        self.rng = numpy.random.default_rng(0)
        y_mod = hmm.observe_float.Gauss(mu_1.copy(), var_1.copy(), self.rng)
        self.model_1_1 = hmm.extensions.HMM(p_s0, p_s0, p_ss, y_mod)
        self.model_2_4 = hmm.extensions.HMM(
            p_s0, p_s0, p_ss,
            hmm.observe_float.Gauss(mu_1 * 2, var_1 * 4, self.rng))
        _, y_train = self.model_1_1.simulate(100)
        self.y_train = np.array(y_train, np.float64).reshape((-1,))

    def test_decode(self):
        self.model_1_1.decode((self.y_train,))

    def test_train(self):
        self.model_2_4.y_mod.observe((self.y_train,))
        self.model_2_4.train((self.y_train,), n_iterations=15)

    def test_str(self):
        self.assertTrue(isinstance(self.model_1_1.y_mod.__str__(), str))


if __name__ == "__main__":
    numpy.testing.run_module_suite()

# --------------------------------
# Local Variables:
# mode: python
# End:
