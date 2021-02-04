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
import hmm.extensions

n_states = 6
model_py_state = scipy.linalg.circulant([0.4, 0, 0, 0, 0.3, 0.3])
p_state2state = scipy.linalg.circulant([0, 0, 0, 0, 0.5, 0.5])
bundle2state = {0: [0, 1, 2], 1: [3], 2: [4, 5]}
p_state_initial = numpy.ones(n_states) / n_states


class TestHMM(unittest.TestCase):
    """ Test hmm.extensions.HMM
    """

    def setUp(self):
        y_class = hmm.scalar.Observation
        p_ys = hmm.base.Prob(model_py_state.copy())
        y_class_parameters = {'model_py_state': p_ys}
        rng = numpy.random.default_rng(0)

        self.observation_args = {
            'y_class_parameters': y_class_parameters,
            'y_class': y_class,
            'bundle2state': bundle2state
        }

        self.observation_class = hmm.extensions.Observation_with_bundles
        self.hmm_class = hmm.extensions.HMM
        self.hmm = self.new_hmm()
        _, observations = self.hmm.simulate(1500)
        self.y = [observations, observations, observations]

    def new_hmm(self):
        rng = numpy.random.default_rng(0)
        observation_model = self.observation_class(self.observation_args, rng)
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
        log_like = self.hmm.multi_train(self.y, n_iter=10, display=True)
        # Check that log likelihood increases montonically
        for i in range(1, len(log_like)):
            self.assertTrue(log_like[i - 1] < log_like[i])
        # Check that trained model is close to true model
        numpy.testing.assert_allclose(
            self.hmm.y_mod.y_mod.model_py_state.values(),
            model_py_state,
            atol=0.1)
        numpy.testing.assert_allclose(self.hmm.p_state2state.values(),
                                      p_state2state,
                                      atol=0.1)


class TestObservation_with_bundles(unittest.TestCase):
    """ Test hmm.extensions.Observation_with_bundles
    """

    def setUp(self):
        self.y_class = hmm.scalar.Observation
        p_ys = hmm.base.Prob(model_py_state.copy())
        self.y_class_parameters = {'model_py_state': p_ys}
        self.bundle2state = bundle2state
        self.rng = numpy.random.default_rng(0)

        self.args = {
            'y_class_parameters': self.y_class_parameters,
            'y_class': self.y_class,
            'bundle2state': self.bundle2state
        }

        self.Observation_with_bundles = hmm.extensions.Observation_with_bundles(
            self.args, self.rng)

        outs = [
            self.Observation_with_bundles.random_out(s)
            for s in range(len(p_ys))
        ]
        bundles = [out[0] for out in outs]
        ys = [out[1] for out in outs]
        self.data = [
            hmm.extensions.Bundle_segment(bundles, ys) for x in range(3)
        ]

    def test_init(self):
        hmm.extensions.Observation_with_bundles(self.args, self.rng)

    def test_observe(self):
        self.assertTrue(self.Observation_with_bundles.observe(self.data) == 18)

    def test_calculate(self):
        self.Observation_with_bundles.observe(self.data)
        result = self.Observation_with_bundles.calculate()
        self.assertTrue(result.min() == 0)
        self.assertTrue(result.max() > .35)

    def test_reestimate(self):
        self.Observation_with_bundles.observe(self.data)
        w = self.Observation_with_bundles.calculate()
        self.Observation_with_bundles.reestimate(w)
        result = self.Observation_with_bundles.y_mod.model_py_state
        self.assertTrue(result.min() == 0)
        self.assertTrue(result.max() == 1.0)


if __name__ == "__main__":
    numpy.testing.run_module_suite()

# --------------------------------
# Local Variables:
# mode: python
# End:
