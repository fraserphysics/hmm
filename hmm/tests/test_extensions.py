"""test_base.py Tests hmm.base

$ python -m pytest hmm/tests/test_base.py

"""

import unittest

import numpy as np
import numpy.testing
import numpy.random

import scipy.linalg

import hmm.extensions
import hmm.base


class TestObservations(unittest.TestCase):
    """ Test Observation in modules extensions and base
    """

    def setUp(self):
        self.numpy_rng = numpy.random.default_rng(0)  # 0 is seed
        p_ys = hmm.base.Prob(numpy.array([[0, 1], [1, 1], [1, 3.0]]))
        p_ys.normalize()
        n = 20
        y = np.empty(n, dtype=np.int32)
        for i in range(n):
            y[i] = (i + i % 2 + i % 3 + i % 5) % 2
        self.y = y
        self.y64 = np.array(y, dtype=np.int64)
        self.w = np.array(20 * [0, 0, 1.0]).reshape((n, 3))
        self.w[0, :] = [1, 0, 0]
        self.w[3, :] = [0, 1, 0]
        self.ys = (y[5:], y[3:7], y[:4])
        self.y_mod_extensions = hmm.extensions.Observation(
            p_ys.copy(), self.numpy_rng)
        self.y_mod_base = hmm.base.Observation(p_ys.copy(), self.numpy_rng)
        # discrete model and observations from extensions
        self.y_mod_y_extensions = (self.y_mod_extensions, (self.y64,))
        # discrete model and observations from base
        self.y_mod_y_base = (self.y_mod_base, self.y64)

    def calc(self, y_mod, y):
        y_mod.observe(y)
        py = y_mod.calculate()[2:4]
        numpy.testing.assert_almost_equal(py, [[0, 0.5, 0.25], [1, 0.5, 0.75]])

    def test_calc(self):
        for y_mod, y in (self.y_mod_y_extensions, self.y_mod_y_base):
            self.calc(y_mod, y)

    def test_join(self):
        self.y_mod_extensions.observe(self.ys)
        numpy.testing.assert_equal(self.y_mod_extensions.t_seg, [0, 15, 19, 23])

    def reestimate(self, y_mod, y):
        y_mod.observe(y)
        y_mod.calculate()
        y_mod.reestimate(self.w)
        numpy.testing.assert_almost_equal([[1, 0], [0, 1], [5 / 9, 4 / 9]],
                                          y_mod._py_state.values())

    def test_reestimate(self):
        for y_mod, y in (self.y_mod_y_extensions, self.y_mod_y_base):
            self.reestimate(y_mod, y)

    def test_str(self):
        self.assertTrue(isinstance(self.y_mod_extensions.__str__(), str))

# Todo: Eliminate globals?
n_states = 6
_py_state = scipy.linalg.circulant([0.4, 0, 0, 0, 0.3, 0.3])
p_state2state = scipy.linalg.circulant([0, 0, 0, 0, 0.5, 0.5])
bundle2state = {0: [0, 1, 2], 1: [3], 2: [4, 5]}
p_state_initial = numpy.ones(n_states) / n_states


class TestHMM(unittest.TestCase):
    """ Test hmm.extensions.HMM
    """

    def setUp(self):
        self.y_class = hmm.extensions.Observation
        self.p_ys = hmm.base.Prob(_py_state.copy())
        self.rng = numpy.random.default_rng(0)
        self.observation_class = hmm.extensions.Observation_with_bundles
        self.bundle2state = bundle2state
        self.hmm_class = hmm.extensions.HMM
        self.hmm = self.new_hmm()
        _, observations = self.hmm.simulate(1000)
        self.y = [observations] * 5

    def new_hmm(self):
        rng = numpy.random.default_rng(0)
        observation_model = self.observation_class([self.p_ys], self.y_class,
                                                   self.bundle2state, rng)
        hmm = self.hmm_class(
            p_state_initial.copy(),  # Initial distribution of states
            p_state_initial.copy(),  # Stationary distribution of states
            p_state2state.copy(),  # State transition probabilities
            observation_model,
            rng=rng,
        )
        return hmm

    def test_state_simulate(self):
        result = self.hmm.state_simulate(10)

    def test_simulate(self):
        n = 10
        result = self.hmm.simulate(n)
        self.assertTrue(len(result[0]) == n)
        self.assertTrue(len(result[1].bundles) == n)

    def test_str(self):
        self.assertTrue(isinstance(self.hmm.__str__(), str))

    def test_multi_train(self):
        """ Test training
        """
        log_like = self.hmm.multi_train(self.y, n_iterations=10, display=False)
        # Check that log likelihood increases montonically
        for i in range(1, len(log_like)):
            self.assertTrue(
                log_like[i - 1] < log_like[i] + 1e-14)  # Todo: fudge?
        # Check that trained model is close to true model
        numpy.testing.assert_allclose(self.hmm.y_mod.y_mod._py_state.values(),
                                      _py_state,
                                      atol=0.15)
        numpy.testing.assert_allclose(self.hmm.p_state2state.values(),
                                      p_state2state,
                                      atol=0.15)


class TestObservation_with_bundles(unittest.TestCase):
    """ Test hmm.extensions.Observation_with_bundles
    """

    def setUp(self):
        self.y_class = hmm.extensions.Observation
        self.p_ys = hmm.base.Prob(_py_state.copy())
        self.bundle2state = bundle2state
        self.rng = numpy.random.default_rng(0)

        self.Observation_with_bundles = hmm.extensions.Observation_with_bundles(
            [self.p_ys], self.y_class, self.bundle2state, self.rng)

        outs = [
            self.Observation_with_bundles.random_out(s)
            for s in range(len(self.p_ys))
        ]
        bundles = [out[0] for out in outs]
        ys = [out[1] for out in outs]
        self.data = [
            hmm.extensions.Bundle_segment(bundles, ys) for x in range(3)
        ]

    def test_init(self):
        hmm.extensions.Observation_with_bundles([self.p_ys], self.y_class,
                                                self.bundle2state, self.rng)

    def test_observe(self):
        t_seg = self.Observation_with_bundles.observe(self.data)
        numpy.testing.assert_equal(t_seg, numpy.array([0, 6, 12, 18]))

    def test_calculate(self):
        self.Observation_with_bundles.observe(self.data)
        result = self.Observation_with_bundles.calculate()
        self.assertTrue(result.min() == 0)
        self.assertTrue(result.max() > .35)

    def test_reestimate(self):
        self.Observation_with_bundles.observe(self.data)
        w = self.Observation_with_bundles.calculate()
        self.Observation_with_bundles.reestimate(w)
        result = self.Observation_with_bundles.y_mod._py_state
        self.assertTrue(result.min() == 0)
        self.assertTrue(result.max() == 1.0)


if __name__ == "__main__":
    numpy.testing.run_module_suite()

# --------------------------------
# Local Variables:
# mode: python
# End:
