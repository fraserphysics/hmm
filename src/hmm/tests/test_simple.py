"""test_simple.py Tests hmm.simple

hmm.simple.Observation is tested with hmm.base.IntegerObservation in
test_base.py

$ python -m pytest hmm/tests/test_simple.py

"""

import unittest

import numpy
import numpy.testing
import numpy.random

import scipy.linalg  # type: ignore

import hmm.simple


class BaseClass(unittest.TestCase):
    """Holds common values and methods used by other test classes.

    Because this class has no method names that start with "test",
    unittest will not discover and run any of the methods.  However
    the methods of this class can be assigned to names that start with
    "test" in subclasses where they will be discovered and run.

    """
    # Values here can be accessed by instances of subclasses as, eg,
    # self.n_states.

    n_states = 6
    n_times = 1000
    p_state_initial = numpy.ones(n_states) / float(n_states)
    p_state_time_average = p_state_initial
    p_state2state = scipy.linalg.circulant([0, 0, 0, 0, 0.5, 0.5])
    _py_state = scipy.linalg.circulant([0.4, 0, 0, 0, 0.3, 0.3])

    # Because of the decorator, setUpClass is called with the class,
    # not an instance as the first argument, hence it is customarily
    # named cls.  It is called once while the method named setUp gets
    # called before each test.
    @classmethod
    def setUpClass(cls):
        """ This code is executed once per test class
        """
        # cls._py_state = scipy.linalg.circulant([0.4, 0, 0, 0, 0.3, 0.3])
        cls.mask = numpy.ones((cls.n_times, cls.n_states), bool)
        for t in range(cls.n_times):
            cls.mask[t, t % cls.n_states] = False

    def initialize(self):
        # pylint: disable = attribute-defined-outside-init
        self.rng = numpy.random.default_rng(0)
        self.simple_observation = hmm.simple.Observation(
            self._py_state.copy(), self.rng)
        self.simple_hmm = hmm.simple.HMM(
            self.p_state_initial.copy(),  # Initial distribution of states
            self.p_state_time_average.copy(
            ),  # Stationary distribution of states
            self.p_state2state.copy(),  # State transition probabilities
            self.simple_observation,
            rng=self.rng,
        )


class TestHMM(BaseClass):
    """ Test base.HMM
    """

    def setUp(self):
        super().initialize()
        self.mods = (self.simple_hmm,
                    )  # More mods after getting models in C built
        self.s, y = self.simple_hmm.simulate(self.n_times)
        self.y = numpy.array(y, numpy.int32).reshape((-1))

    def test_state_simulate(self):
        result1 = self.simple_hmm.state_simulate(self.n_times)
        result2 = self.simple_hmm.state_simulate(self.n_times, self.mask)
        for result in (result1, result2):
            self.assertTrue(len(result) == self.n_times)
            array = numpy.array(result)
            self.assertTrue(array.min() == 0)
            self.assertTrue(array.max() == self.n_states - 1)

    def test_link(self):
        """ Remove link from 0 to itself
        """
        self.simple_hmm.link(0, 0, 0)
        self.assertTrue(self.simple_hmm.p_state2state[0, 0] == 0.0)

    def test_str(self):
        string = self.simple_hmm.__str__()
        self.assertTrue(string.find('with 6 states') == 25)

    def test_unseeded(self):
        """ Exercise initialization of HMM without supplying rng
        """

        mod = hmm.simple.HMM(
            self.p_state_initial.copy(),  # Initial distribution of states
            self.p_state_time_average.copy(
            ),  # Stationary distribution of states
            self.p_state2state.copy(),  # State transition probabilities
            self.simple_observation,
        )
        states, y = mod.simulate(10)  # pylint: disable = unused-variable
        self.assertTrue(len(states) == 10)

    def test_decode(self):
        """
        Check that self.mod gets 70% of the states right
        """
        states = self.simple_hmm.decode(self.y)
        wrong = numpy.where(states != self.s)[0]
        self.assertTrue(len(wrong) < len(self.s) * .3)
        # Check that other models get the same state sequence as self.hmm
        for mod in self.mods[1:]:
            wrong = numpy.where(states != mod.decode(self.y))[0]
            self.assertTrue(len(wrong) == 0)

    def test_train(self):
        """ Test training
        """
        log_like = self.simple_hmm.train(self.y, n_iterations=10, display=True)
        # Check that log likelihood increases montonically
        for i in range(1, len(log_like)):
            self.assertTrue(log_like[i - 1] < log_like[i])
        # Check that trained model is close to true model
        numpy.testing.assert_allclose(self.simple_hmm.y_mod._py_state.values(),
                                      self._py_state,
                                      atol=0.15)
        numpy.testing.assert_allclose(self.simple_hmm.p_state2state.values(),
                                      self.p_state2state,
                                      atol=0.2)
        # Check that other models give results close to self.hmm
        for mod in self.mods[1:]:
            log_like_mod = mod.train(self.y, n_iter=10, display=False)
            numpy.testing.assert_allclose(log_like_mod, log_like)
            numpy.testing.assert_allclose(
                mod.y_mod._py_state.values(),
                self.simple_hmm.y_mod._py_state.values())
            numpy.testing.assert_allclose(
                mod.p_state2state.values(),
                self.simple_hmm.p_state2state.values())


class TestProb(BaseClass):
    """ Tests hmm.simple.Prob
    """

    A_ = numpy.array([[0, 2, 2.0], [2, 2, 4.0], [6, 2, 2.0]])
    B_ = numpy.array([[0, 1], [1, 1], [1, 3.0]])
    C_ = numpy.array([[0, 0, 2.0], [0, 0, 1.0], [6, 0, 0.0]])

    def setUp(self):
        """ This code is run before each test
        """
        self.a = hmm.simple.Prob(self.A_.copy())
        self.b = hmm.simple.Prob(self.B_.copy())
        self.c = hmm.simple.Prob(self.C_.copy())
        self.ms = (self.a, self.b, self.c)
        for m in self.ms:
            m.normalize()

    def test_normalize(self):
        for m in self.ms:
            n_rows, n_columns = m.shape
            for i in range(n_rows):
                s = 0
                for j in range(n_columns):
                    s += m.values()[i, j]
                numpy.testing.assert_almost_equal(1, s)

    def test_assign(self):
        a = self.c.values().sum()
        self.c.assign_col(1, [1, 1, 1])
        numpy.testing.assert_almost_equal(self.c.values().sum(), a + 3)

    def test_likelihoods(self):
        numpy.testing.assert_allclose(
            self.c.likelihoods([0, 1, 2])[2], [1, 1, 0])

    def test_utility(self):
        numpy.testing.assert_almost_equal(
            self.c.utility(self.b.T[0], self.b.T[1]),
            [[0, 0, 0], [0, 0, 0.375], [0.25, 0, 0]])

    def test_step_forward(self):
        b = numpy.array([1., 0.5, 0.75])
        self.a.step_forward(b)
        numpy.testing.assert_almost_equal(b, [0.575, 0.775, 0.9])

    def test_step_back(self):
        b = numpy.array([1., 0.5, 0.75])
        self.a.step_back(b)
        numpy.testing.assert_almost_equal(b, [0.625, 0.75, 0.85])

    def test_values(self):
        numpy.testing.assert_almost_equal(self.c.values(),
                                          [[0, 0, 1], [0, 0, 1], [1, 0, 0]])


# --------------------------------
# Local Variables:
# mode: python
# End:
