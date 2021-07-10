"""test_observe_float.py: T run "$ python -m pytest test_observe_float.py" or
"$ python -m pytest hmm/tests"

"""
# Copyright (c) 2021 Andrew M. Fraser
import unittest

import numpy
import numpy.testing

import scipy.linalg  # type: ignore

import hmm.observe_float
import hmm.base


class Parent(unittest.TestCase):
    """ Don't use this class.  Use subclasses
    """

    # If you use method names for this class that begin with test,
    # they will be called with an instance of class Parent even though
    # the name Parent doesn't begin with Test.

    def setUp(self):

        p_initial_state = [0.67, 0.33]
        p_state2state = [[0.93, 0.07], [0.13, 0.87]]
        self.rng = numpy.random.default_rng(0)
        y_mod_a, y_mod_b = self.make_y_mods()  # pylint: disable = no-member
        self.model_a = hmm.base.HMM(p_initial_state, p_initial_state,
                                    p_state2state, y_mod_a, self.rng)
        self.model_b = hmm.base.HMM(p_initial_state, p_initial_state,
                                    p_state2state, y_mod_b)

        if hasattr(self.model_a.y_mod, 'initialize_out'):
            self.model_a.y_mod.initialize_out()  # For AutoRegressive y_mod
        _, y_train = self.model_a.simulate(100)
        self.y_shape = y_train.shape
        self.y_type = y_train.dtype

        self.y_train = numpy.array(y_train, numpy.float64)

    def _decode(self, sequence_sum):
        """ Called by test_decode in subclasses
        """
        rv = numpy.array(self.model_a.decode((self.y_train,)))
        self.assertTrue(rv.sum() == sequence_sum)

    def _train(self):
        """ Called by test_train in subclasses
        """
        self.model_b.y_mod.observe((self.y_train,))

        # Exercises calculate and reestimate
        rv = numpy.array(
            self.model_b.multi_train((self.y_train,), n_iterations=100))

        difference = rv[1:] - rv[:-1]
        self.assertTrue(difference.min() > -1.0e-15)  # Check monotonic

    def _str(self, reference_string):
        """ Called by test_str in subclasses
        """
        string = self.model_a.y_mod.__str__()
        n_instance = string.find('instance')
        tail = string[n_instance:]
        self.assertTrue(tail == reference_string)


class TestGauss(Parent):

    def make_y_mods(self):
        mu_1 = numpy.array([-1.0, 1.0])
        var_1 = numpy.ones(2)
        y_mod_a = hmm.observe_float.Gauss(mu_1.copy(), var_1.copy(), self.rng)
        y_mod_b = hmm.observe_float.Gauss(mu_1 * 2, var_1 * 4, self.rng)
        return y_mod_a, y_mod_b

    def test_decode(self):
        self._decode(49)

    def test_train(self):
        self._train()

    def test_str(self):
        self._str('instance:\n    mu\n[-1.  1.]\n    variance\n[1. 1.]\n')


class TestGaussMAP(TestGauss):

    def make_y_mods(self):
        mu_1 = numpy.array([-1.0, 1.0])
        var_1 = numpy.ones(2)
        y_mod_a = hmm.observe_float.GaussMAP(mu_1.copy(), var_1.copy(),
                                             self.rng)
        y_mod_b = hmm.observe_float.GaussMAP(mu_1 * 2, var_1 * 4, self.rng)
        return y_mod_a, y_mod_b


class TestMultivariate(Parent):

    def make_y_mods(self):
        mu_1 = numpy.array([[1.0, 1.0], [-1.0, -1.0]])
        mu_2 = numpy.array([[1.0, -1.0], [-1.0, 1.0]])
        var = numpy.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        y_mod_a = hmm.observe_float.MultivariateGaussian(
            mu_1.copy(), var.copy(), self.rng)
        y_mod_b = hmm.observe_float.MultivariateGaussian(
            mu_2.copy(), var.copy(), self.rng)
        return y_mod_a, y_mod_b

    def test_decode(self):
        self._decode(47)

    def test_train(self):
        self._train()

    def test_str(self):
        self._str(
            'instance\nFor state 0:\n inverse_sigma = \n[[1. 0.]\n [0. 1.]]\n mu = [1. 1.] norm = 0.159155\nFor state 1:\n inverse_sigma = \n[[1. 0.]\n [0. 1.]]\n mu = [-1. -1.] norm = 0.159155\n'  # pylint: disable = line-too-long
        )


class TestAutoRegressive(Parent):
    """ AR order: 1
    """

    def make_y_mods(self):
        ar_1 = numpy.array([[0.5], [-0.5]])
        offset_1 = numpy.array([0.5, -0.5])
        ar_2 = numpy.array([[0.85], [-0.85]])
        offset_2 = numpy.array([-0.5, 0.5])
        var = numpy.array([1.0, 1.0])

        y_mod_a = hmm.observe_float.AutoRegressive(ar_1.copy(), offset_1.copy(),
                                                   var.copy(), self.rng)

        y_mod_b = hmm.observe_float.AutoRegressive(ar_2.copy(), offset_2.copy(),
                                                   var.copy(), self.rng)

        return y_mod_a, y_mod_b

    def test_decode(self):
        self._decode(42)

    def test_train(self):
        self._train()

    def test_str(self):
        self._str(
            'instance\nFor state 0:\n variance = \n1.0\n ar_coefficients = [0.5] offset = 0.5 norm = 0.398942\nFor state 1:\n variance = \n1.0\n ar_coefficients = [-0.5] offset = -0.5 norm = 0.398942\n'  # pylint: disable = line-too-long
        )


# --------------------------------
# Local Variables:
# mode: python
# End:
