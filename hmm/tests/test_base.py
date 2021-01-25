"""test_base.py Tests hmm.base

hmm.base.Observation is tested with hmm.scalar.Observation in test_scalar.py
$ python -m pytest hmm/tests/test_base.py

"""

import unittest

import numpy as np
import numpy.testing
import numpy.random

import scipy.linalg

import hmm.scalar
import hmm.base

n_states = 6
n_t = 1000
p_state_initial = np.ones(n_states) / float(n_states)
p_state_time_average = np.ones(n_states) / float(n_states)
p_state2state = scipy.linalg.circulant([0, 0, 0, 0, 0.5, 0.5])
model_py_state = scipy.linalg.circulant([0.4, 0, 0, 0, 0.3, 0.3])


class TestHMM(unittest.TestCase):
    """ Test base.HMM
    """

    def setUp(self):
        self.rng = numpy.random.default_rng(0)
        self.y_model = hmm.base.Observation(model_py_state.copy(), self.rng)
        self.hmm = hmm.base.HMM(
            p_state_initial.copy(),  # Initial distribution of states
            p_state_time_average.copy(),  # Stationary distribution of states
            p_state2state.copy(),  # State transition probabilities
            self.y_model,
            rng=self.rng,
        )
        self.mask = np.ones((n_t, n_states), np.bool)
        for t in range(n_t):
            self.mask[t, t % n_states] = False
        self.mods = (self.hmm,)  # More mods after getting models in C built
        self.s, y = self.hmm.simulate(n_t)
        self.y = np.array(y, np.int32).reshape((-1))

    def test_state_simulate(self):
        self.hmm.state_simulate(n_t)
        self.hmm.state_simulate(n_t, self.mask)

    def test_initialize_y_model(self):
        """ Also exercises self.mod.state_simulate.
        """
        self.hmm.initialize_y_model(self.y)

    def test_link(self):
        """ Remove link from 0 to itself
        """
        self.hmm.link(0, 0, 0)
        self.assertTrue(self.hmm.p_state2state[0, 0] == 0.0)

    def test_str(self):
        self.assertTrue(isinstance(self.hmm.__str__(), str))

    def test_unseeded(self):
        """ Exercise initialization of HMM without supplying rng
        """

        mod = hmm.base.HMM(
            p_state_initial.copy(),  # Initial distribution of states
            p_state_time_average.copy(),  # Stationary distribution of states
            p_state2state.copy(),  # State transition probabilities
            self.y_model,
        )
        states, y = mod.simulate(10)  # pylint: disable = unused-variable
        self.assertTrue(len(states) == 10)

    def test_decode(self):
        """
        Check that self.mod gets 70% of the states right
        """
        states = self.hmm.decode(self.y)
        wrong = np.where(states != self.s)[0]
        self.assertTrue(len(wrong) < 300)
        # Check that other models get the same state sequence as self.hmm
        for mod in self.mods[1:]:
            wrong = np.where(states != mod.decode(self.y))[0]
            self.assertTrue(len(wrong) == 0)

    def test_train(self):
        """ Test training
        """
        log_like = self.hmm.train(self.y, n_iter=10, display=True)
        # Check that log likelihood increases montonically
        for i in range(1, len(log_like)):
            self.assertTrue(log_like[i - 1] < log_like[i])
        # Check that trained model is close to true model
        numpy.testing.assert_allclose(self.hmm.y_mod.model_py_state.values(),
                                      model_py_state,
                                      atol=0.15)
        numpy.testing.assert_allclose(self.hmm.p_state2state.values(),
                                      p_state2state,
                                      atol=0.2)
        # Check that other models give results close to self.hmm
        for mod in self.mods[1:]:
            log_like_mod = mod.train(self.y, n_iter=10, display=False)
            numpy.testing.assert_allclose(log_like_mod, log_like)
            numpy.testing.assert_allclose(
                mod.y_mod.model_py_state.values(),
                self.hmm.y_mod.model_py_state.values())
            numpy.testing.assert_allclose(mod.p_state2state.values(),
                                          self.hmm.p_state2state.values())

    # todo delete
    def multi_train(self, mod):
        """ Test training on multiple sequences.
        """
        ys = []
        for i in [1, 2, 0, 4, 3]:
            ys.append([x[200 * i:200 * (i + 1)] for x in self.y])
        log_like = mod.multi_train(ys, n_iter=10, display=False)
        for i in range(1, len(log_like)):
            self.assertTrue(log_like[i - 1] < log_like[i])
        numpy.testing.assert_allclose(mod.y_mod.model_py_state.values(),
                                      model_py_state,
                                      atol=0.14)
        numpy.testing.assert_allclose(mod.p_state2state.values(),
                                      p_state2state,
                                      atol=0.17)

    def foo_test_multi_train(self):
        for mod in self.mods:
            self.multi_train(mod)


A_ = numpy.array([[0, 2, 2.0], [2, 2, 4.0], [6, 2, 2.0]])
B_ = numpy.array([[0, 1], [1, 1], [1, 3.0]])
C_ = numpy.array([[0, 0, 2.0], [0, 0, 1.0], [6, 0, 0.0]])


class TestProb(unittest.TestCase):
    """ Tests hmm.base.Prob
    """

    def setUp(self):
        self.a = hmm.base.Prob(A_.copy())
        self.b = hmm.base.Prob(B_.copy())
        self.c = hmm.base.Prob(C_.copy())
        self.ms = (self.a, self.b, self.c)
        for m in self.ms:
            m.normalize()

    def test_normalize(self):
        for m_ in self.ms:
            m, n = m_.shape
            for i in range(m):
                s = 0
                for j in range(n):
                    s += m_.values()[i, j]
                numpy.testing.assert_almost_equal(1, s)

    def assign(self, m):
        a = m.values().sum()
        m.assign_col(1, [1, 1, 1])
        numpy.testing.assert_almost_equal(m.values().sum(), a + 3)

    def test_assign(self):
        for m in (self.c,):
            self.assign(m)

    def likelihoods(self, m):
        numpy.testing.assert_allclose(m.likelihoods([0, 1, 2])[2], [1, 1, 0])

    def test_likelihoods(self):
        for m in (self.c,):
            self.likelihoods(m)

    def cost(self, m):
        numpy.testing.assert_almost_equal(m.cost(
            self.b.T[0], self.b.T[1]), [[0, 0, 0], [0, 0, 0.375], [0.25, 0, 0]])

    def test_cost(self):
        for m in (self.c,):
            self.cost(m)

    def inplace_elementwise_multiply(self, m):
        m *= self.a
        numpy.testing.assert_almost_equal(
            m.values(), [[0, 0, 0.5], [0, 0, 0.5], [0.6, 0, 0]])

    def test_inplace_elementwise_multiply(self):
        for m in (self.c,):
            self.inplace_elementwise_multiply(m)

    def step_forward(self, m):
        b = self.b.T[1].copy()
        m.step_forward(b)
        numpy.testing.assert_almost_equal(b, [0.575, 0.775, 0.9])

    def test_step_forward(self):
        for m in (self.a,):
            self.step_forward(m)

    def step_back(self, m):
        b = self.b.T[1].copy()
        m.step_back(b)
        numpy.testing.assert_almost_equal(b, [0.625, 0.75, 0.85])

    def test_step_back(self):
        for m in (self.a,):
            self.step_back(m)

    def values(self, m):
        numpy.testing.assert_almost_equal(m.values(),
                                          [[0, 0, 1], [0, 0, 1], [1, 0, 0]])

    def test_values(self):
        for m in (self.c,):
            self.values(m)


if __name__ == "__main__":
    numpy.testing.run_module_suite()

# --------------------------------
# Local Variables:
# mode: python
# End:
