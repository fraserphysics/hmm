"""test_base.py Tests hmm.base

$ python -m pytest hmm/tests/test_base.py or $ py.test --pdb hmm/tests/test_base.py

"""

import unittest

import numpy
import numpy.testing
import numpy.random

import scipy.linalg  # type: ignore

import hmm.base
import hmm.simple
import tests.test_simple


class TestObservations(unittest.TestCase):
    """ Test Observation in modules simple and base
    """

    def setUp(self):
        self.numpy_rng = numpy.random.default_rng(0)  # 0 is seed
        p_ys = hmm.simple.Prob(numpy.array([[0, 1], [1, 1], [1, 3.0]]))
        p_ys.normalize()
        n = 20
        y = numpy.empty(n, dtype=numpy.int32)
        for i in range(n):
            y[i] = (i + i % 2 + i % 3 + i % 5) % 2
        self.y = y
        self.y64 = numpy.array(y, dtype=numpy.int64)
        self.w = numpy.array(20 * [0, 0, 1.0]).reshape((n, 3))
        self.w[0, :] = [1, 0, 0]
        self.w[3, :] = [0, 1, 0]
        self.ys = (y[5:], y[3:7], y[:4])
        self.y_mod_base = hmm.base.IntegerObservation(p_ys.copy(),
                                                      self.numpy_rng)
        self.y_mod_simple = hmm.simple.Observation(p_ys.copy(), self.numpy_rng)
        # discrete model and observations from base
        self.y_mod_y_base = (self.y_mod_base, (self.y64,))
        # discrete model and observations from simple
        self.y_mod_y_simple = (self.y_mod_simple, self.y64)

    def calc(self, y_mod, y):
        y_mod.observe(y)
        py = y_mod.calculate()[2:4]
        numpy.testing.assert_almost_equal(py, [[0, 0.5, 0.25], [1, 0.5, 0.75]])

    def test_calc(self):
        for y_mod, y in (self.y_mod_y_base, self.y_mod_y_simple):
            self.calc(y_mod, y)

    def test_join(self):
        self.y_mod_base.observe(self.ys)
        numpy.testing.assert_equal(self.y_mod_base.t_seg, [0, 15, 19, 23])

    def reestimate(self, y_mod, y):
        y_mod.observe(y)
        y_mod.calculate()
        y_mod.reestimate(self.w)
        numpy.testing.assert_almost_equal([[1, 0], [0, 1], [5 / 9, 4 / 9]],
                                          y_mod._py_state.values())

    def test_reestimate(self):
        for y_mod, y in (self.y_mod_y_base, self.y_mod_y_simple):
            self.reestimate(y_mod, y)

    def test_str(self):
        self.assertTrue(isinstance(self.y_mod_base.__str__(), str))


class BaseClass(tests.test_simple.BaseClass):
    bundle2state = {0: [0, 1, 2], 1: [3], 2: [4, 5]}


class TestHMM(BaseClass):
    """ Test hmm.base.HMM
    """

    def setUp(self):
        self.y_class = hmm.base.IntegerObservation
        self.p_ys = hmm.simple.Prob(self._py_state.copy())
        self.rng = numpy.random.default_rng(0)

        # Setup for test_initialize_y_model with simple_observation
        simple_observation = hmm.base.IntegerObservation(self.p_ys, self.rng)
        self.simple_hmm = hmm.base.HMM(
            self.p_state_initial.copy(),  # Initial distribution of states
            self.p_state_initial.copy(),  # Stationary distribution of states
            self.p_state2state.copy(),  # State transition probabilities
            simple_observation,
            rng=self.rng,
        )
        _, observations = self.simple_hmm.simulate(1000)
        self.simple_y = [observations] * 5

        # Setup for other tests with bundle_observation
        bundle_observation = hmm.base.Observation_with_bundles(
            simple_observation, self.bundle2state, self.rng)
        self.base_hmm = hmm.base.HMM(
            self.p_state_initial.copy(),  # Initial distribution of states
            self.p_state_initial.copy(),  # Stationary distribution of states
            self.p_state2state.copy(),  # State transition probabilities
            bundle_observation,
            rng=self.rng,
        )
        _, observations = self.base_hmm.simulate(1000)
        self.y = [observations] * 5

    def test_initialize_y_model(self):
        """ Also exercises self.mod.state_simulate.
        """
        # Need .copy() because initialize_y_model modifies _py_state
        difference = self.simple_hmm.y_mod._py_state.copy(
        ) - self.simple_hmm.initialize_y_model(self.simple_y)._py_state
        self.assertTrue(difference.max() > 0.01)
        zero = difference.sum(axis=1)
        self.assertTrue(zero.max() < 1e-9)  # Rows of each should sum to one

    def test_state_simulate(self):
        self.base_hmm.state_simulate(10)

    def test_simulate(self):
        n = 10
        result = self.base_hmm.simulate(n)
        self.assertTrue(len(result[0]) == n)
        self.assertTrue(len(result[1].bundles) == n)

    def test_str(self):
        self.assertTrue(isinstance(self.base_hmm.__str__(), str))

    def test_t_skip(self):
        self.base_hmm.y_mod.observe(self.y[:1])
        self.base_hmm.state_likelihood = self.base_hmm.y_mod.calculate()
        n_times = self.base_hmm.y_mod.n_times
        n_states = self.base_hmm.n_states
        self.base_hmm.alpha = numpy.empty((n_times, n_states))
        self.base_hmm.gamma_inv = numpy.empty((n_times,))
        self.assertTrue(self.base_hmm.forward(t_skip=5) > -2000.0)

    def test_multi_train(self):
        """ Test training
        """
        log_like = self.base_hmm.multi_train(self.y,
                                             n_iterations=10,
                                             display=False)
        # Check that log likelihood increases montonically
        for i in range(1, len(log_like)):
            self.assertTrue(
                log_like[i - 1] < log_like[i] + 1e-14)  # Todo: fudge?
        # Check that trained model is close to true model
        numpy.testing.assert_allclose(
            self.base_hmm.y_mod.underlying_model._py_state.values(),
            self._py_state,
            atol=0.15)
        numpy.testing.assert_allclose(self.base_hmm.p_state2state.values(),
                                      self.p_state2state,
                                      atol=0.15)


class TestObservation_with_bundles(BaseClass):
    """ Test hmm.base.Observation_with_bundles
    """

    def setUp(self):
        self.y_class = hmm.base.IntegerObservation
        self.p_ys = hmm.simple.Prob(self._py_state.copy())
        self.rng = numpy.random.default_rng(0)

        self.Observation_with_bundles = hmm.base.Observation_with_bundles(
            self.y_class(self.p_ys, self.rng), self.bundle2state, self.rng)

        # outs[t] = (bundle[t], y[t])
        outs = [
            self.Observation_with_bundles.random_out(s)
            for s in range(len(self.p_ys))
        ]

        bundles = [out[0] for out in outs]
        ys = [out[1] for out in outs]
        self.data = [hmm.base.Bundle_segment(bundles, ys) for x in range(3)]

    def test_bundle_segment_str(self):
        self.assertTrue(
            str(self.data[0]) ==
            'y values:[1, 1, 2, 3, 5, 5]\nbundle values:[0, 0, 0, 1, 2, 2]\n')

    def test_bundle_segment_len(self):
        self.assertTrue(len(self.data[0]) == 6)

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
        result = self.Observation_with_bundles.underlying_model._py_state
        self.assertTrue(result.min() == 0)
        self.assertTrue(result.max() == 1.0)


# --------------------------------
# Local Variables:
# mode: python
# End:
