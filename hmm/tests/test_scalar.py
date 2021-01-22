"""Run with "$ python -m pytest test_scalar.py" or "$ python -m
pytest hmm/tests"

"""
# Copyright (c) 2021 Andrew M. Fraser
import unittest

import numpy as np
import numpy.testing

import scipy.linalg

import hmm.scalar
import hmm.base

A_ = numpy.array([[0, 2, 2.0], [2, 2, 4.0], [6, 2, 2.0]])
B_ = numpy.array([[0, 1], [1, 1], [1, 3.0]])
C_ = numpy.array([[0, 0, 2.0], [0, 0, 1.0], [6, 0, 0.0]])


class TestFunctions(unittest.TestCase):
    """ Test functions not in classes
    """

    def setUp(self):
        self.rng = numpy.random.default_rng(0)

    def test_make_random(self):
        shape = (5, 4)
        conditional_dist = hmm.scalar.make_random(shape, self.rng)
        self.assertTrue(shape == conditional_dist.shape)


class TestObservations(unittest.TestCase):
    """ Test Observation in modules scalar and base
    """

    def setUp(self):
        self.numpy_rng = numpy.random.default_rng(0)  # 0 is seed
        p_ys = hmm.base.Prob(B_.copy())
        p_ys.normalize()
        n = 20
        y = np.empty(n, dtype=np.int32)
        for i in range(n):
            y[i] = (i + i % 2 + i % 3 + i % 5) % 2
        self.y = y
        self.w = np.array(20 * [0, 0, 1.0]).reshape((n, 3))
        self.w[0, :] = [1, 0, 0]
        self.w[3, :] = [0, 1, 0]
        self.ys = (y[5:], y[3:7], y[:4])
        self.y_mod_scalar = hmm.scalar.Observation({'model_py_state': p_ys},
                                                   self.numpy_rng)
        self.y_mod_base = hmm.base.Observation(p_ys, self.numpy_rng)
        # discrete model and observations from scalar
        self.y_mod_y_scalar = (self.y_mod_scalar, (self.y,))
        # discrete model and observations from base
        self.y_mod_y_base = (self.y_mod_base, self.y)

    def calc(self, y_mod, y):
        y_mod.observe(y)
        py = y_mod.calculate()[2:4]
        numpy.testing.assert_almost_equal(py, [[0, 0.5, 0.25], [1, 0.5, 0.75]])

    def test_calc(self):
        for y_mod, y in (self.y_mod_y_scalar, self.y_mod_y_base):
            self.calc(y_mod, y)

    def test_join(self):
        self.y_mod_scalar.observe(self.ys)
        numpy.testing.assert_equal(self.y_mod_scalar.t_seg, [0, 15, 19, 23])

    def reestimate(self, y_mod, y):
        y_mod.observe(y)
        y_mod.calculate()
        y_mod.reestimate(self.w)
        numpy.testing.assert_almost_equal([[1, 0], [0, 1], [5 / 9, 4 / 9]],
                                          y_mod.model_py_state.values())

    def test_reestimate(self):
        for y_mod, y in (self.y_mod_y_scalar, self.y_mod_y_base):
            self.reestimate(y_mod, y)

    def test_str(self):
        self.assertTrue(isinstance(self.y_mod_scalar.__str__(), str))


class TestGauss(unittest.TestCase):
    """ Test hmm.scalar.Gauss TODO: improve these tests.
    """

    def setUp(self):

        p_s0 = [0.67, 0.33]
        p_ss = [[0.93, 0.07], [0.13, 0.87]]
        mu_1 = np.array([-1.0, 1.0])
        var_1 = np.ones(2)
        self.rng = numpy.random.default_rng(0)
        y_mod = hmm.scalar.Gauss({
            'mu': mu_1.copy(),
            'var': var_1.copy(),
        }, self.rng)
        self.model_1_1 = hmm.base.HMM(p_s0, p_s0, p_ss, y_mod)
        self.model_2_4 = hmm.base.HMM(
            p_s0, p_s0, p_ss,
            hmm.scalar.Gauss({
                'mu': mu_1 * 2,
                'var': var_1 * 4
            }, self.rng))
        _, y_train = self.model_1_1.simulate(100)
        self.y_train = np.array(y_train, np.float64).reshape((-1,))

    def test_decode(self):
        self.model_1_1.decode((self.y_train,))

    def test_train(self):
        self.model_2_4.y_mod.observe((self.y_train,), n_y=100)
        self.model_2_4.train((self.y_train,), n_iter=15)

    def test_str(self):
        self.assertTrue(isinstance(self.model_1_1.y_mod.__str__(), str))


if __name__ == "__main__":
    numpy.testing.run_module_suite()

# --------------------------------
# Local Variables:
# mode: python
# End:
