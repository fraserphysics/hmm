""" extensions.py: Extensions of base.py that are necessary for some applications

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import typing  # For type hints

import numpy
import numpy.random

import hmm.base


class HMM(hmm.base.HMM):

    def multi_train(self: HMM, ys, n_iter: int, display=True):
        """Train on more than one independent sequence of observations

        Args:
            ys: Measured observations in format appropriate for self.y_mod
            n_iter: The number of iterations to execute

        Returns:
            List of log likelihood per observation for each iteration

        The differences from base.HMM: 1. More than one independent
        observation sequence.  2. The structure of observations is not specified.

        Todo: What is assumed about the initial state probabilities?

        """

        log_like_list = []
        self.n_times = self.y_mod.observe(ys)
        # t_seg is not returned by observe() to keep base.train simple
        t_seg = self.y_mod.t_seg

        # The times in the ith observation sequence satisfy t_seg[i]
        # \leq t < t_seg[i+1]
        assert self.n_times > 1
        assert self.n_times == t_seg[-1]
        assert t_seg[0] == 0
        n_seg = len(t_seg) - 1
        alpha_all = numpy.empty((self.n_times, self.n_states))
        beta_all = numpy.empty((self.n_times, self.n_states))
        gamma_inv_all = numpy.empty((self.n_times,))
        p_state_initial_all = numpy.empty((n_seg, self.n_states))
        # State probabilities at the beginning of each segment
        for seg in range(n_seg):
            p_state_initial_all[seg, :] = self.p_state_initial.copy()
        for iteration in range(n_iter):
            if display:
                print("i=%d: " % iteration, end="")
            log_like_iteration = 0.0
            # Both forward() and backward() should operate on each
            # training segment and put the results in the
            # corresponding segement of the the alpha, beta and gamma
            # arrays.
            p_y_all = self.y_mod.calculate()
            for seg in range(n_seg):
                # Set up self to run forward/backward on a segment of y
                self.n_times = t_seg[seg + 1] - t_seg[seg]
                self.alpha = alpha_all[t_seg[seg]:t_seg[seg + 1], :]
                self.beta = beta_all[t_seg[seg]:t_seg[seg + 1], :]
                self.p_y_by_state = p_y_all[t_seg[seg]:t_seg[seg + 1]]
                self.gamma_inv = gamma_inv_all[t_seg[seg]:t_seg[seg + 1]]
                self.p_state_initial = p_state_initial_all[seg, :]

                # Run forward/backward to calculate alpha and beta
                # values for a segment of y
                log_like = self.forward()  # Log Likelihood
                if display:
                    print("L[%d]=%7.4f " % (seg, log_like / self.n_times), end="")
                log_like_iteration += log_like
                self.backward()
                p_state_initial_all[seg, :] = self.alpha[0] * self.beta[0]
                self.gamma_inv[
                    0] = -1  # Don't fit state transitions between segments

            log_like_list.append(log_like_iteration / self.n_times)
            if iteration > 0 and log_like_list[iteration -
                                               1] >= log_like_list[iteration]:
                print("""
WARNING training is not monotonic: avg[{0}]={1} and avg[{2}]={3}
""".format(iteration - 1, log_like_list[iteration - 1], iteration,
                log_like_list[iteration]))
            if display:
                print("avg={0:10.7}f".format(log_like_list[-1]))

            # Associate all of the alpha and beta segments with self
            # and reestimate()
            self.alpha = alpha_all
            self.beta = beta_all
            self.gamma_inv = gamma_inv_all
            self.p_y_by_state = p_y_all
            self.reestimate()
        self.p_state_initial[:] = p_state_initial_all.sum(axis=0)
        self.p_state_initial /= self.p_state_initial.sum()
        return log_like_list

    def simulate(self: HMM, n: int) -> Bundle_segment:
        states, outs = super().simulate(n)
        bundles = []
        y = []
        for out in outs:
            bundles.append(out[0])
            y.append(out[1])
        return states, Bundle_segment(bundles, y)


class Bundle_segment:
    """Each instance is a pair of time series.  One of bundle ids and one
    of observations.

    Args:
        bundles: Tags for each time
        y: Observations for each time

    """

    def __init__(self: Bundle_segment, bundles, y):
        self.bundles = bundles
        self.y = y

    def __len__(self: Bundle_segment) -> int:
        return len(self.bundles)

    def __str__(self: Bundle_segment) -> str:
        return """y values:{0}

bundle values:{1}
""".format(self.y, self.bundles)


class Observation_with_bundles(hmm.scalar.Observation):
    """Represents likelihood of states given observations by masking an
    underlying likelihood model.

    Args:
        parameters: A dict {'y_class_parameters': dict, # parameters for y_class,
            'y_class': class, # Observation class without bundles,
            'bundle2state':dict, # Defines bundles of states}
        rng: A numpy.random.Generator for simulation

    The parameter bundle2state is a dict that defines an
    exhaustive partition of the states.  bundle2state[bundle_id] is a
    list of the states in bundle.

    """
    _parameter_keys = set('y_class_parameters y_class bundle2state'.split())

    def __init__(  # pylint: disable = super-init-not-called
            self: Observation_with_bundles, parameters: dict,
            rng: numpy.random.Generator):
        super().__init__(parameters, rng)  # Calls self._normalize
        self.y_mod = self.y_class(self.y_class_parameters, rng)

    def _normalize(self: Observation_with_bundles):
        """Called by super().__init__.  Set up:

        self.bundle_and_state, a boolean numpy.ndarray so that
            self.bundle_and_state[i, j] is True iff state j is in
            bundle i.

        self.state2bundle, an int numpy.ndarray so that if state i is
            in bundle j them self.state2bundle[i] = j.

        """

        self.n_bundle = len(self.bundle2state)

        # Find out how many states there are and ensure that each
        # state is in only one bundle.
        states = set()
        for x in self.bundle2state.values():
            # Assert that no state is in more than one bundle.  Think
            # before relaxing this.
            assert states & set(x) == set()
            states = states | set(x)
        n_states = len(states)
        # Ensure that states is a set of sequential integers [0,n_states)
        assert states == set(range(n_states))

        # bundle_and_state[bundle_id, state_id] is true iff state \in bundle.
        self.bundle_and_state = numpy.zeros((self.n_bundle, n_states),
                                            numpy.bool)

        # state2bundle[state_id] = bundle_id for bundle that contains state
        self.state2bundle = numpy.ones(n_states, dtype=numpy.int32) * -1

        for bundle_id, states in self.bundle2state.items():
            for state_id in states:
                self.bundle_and_state[bundle_id, state_id] = True
                # Ensure that no state is in more than one bundle
                assert self.state2bundle[state_id] == -1
                self.state2bundle[state_id] = bundle_id

        return n_states

    def observe(self: Observation_with_bundles,
                bundle_segment_list: list) -> int:
        """Attach observations to self as a single Bundle_segment

        Args:
            bundle_segment_list: List of Bundle_segments

        Side effects: 1. Assign self.t_seg; 2. Call self.y_mod.observe
        to attach observations stripped of bundle tags to self.y_mod.

        """
        n_times = super().observe(bundle_segment_list)  # Assign self._y
        _n_times = self.y_mod.observe([self._y.y])
        assert n_times == _n_times
        return n_times

    def _concatenate(self: Observation_with_bundles,
                     bundle_segment_list: list) -> Bundle_segment:
        """ Create a single Bundle_segment from a list of segments

        Args:
            bundle_segment_list: Each element is a Bundle_segment
        """
        bundles = []
        ys = []
        for segment in bundle_segment_list:
            bundles.append(segment.bundles)
            ys.append(segment.y)

        def concatenate(list_of_lists):
            """Concatenate a list of lists
            """
            return [item for sublist in list_of_lists for item in sublist]

        return Bundle_segment(concatenate(bundles), concatenate(ys))

    def random_out(self: Observation_with_bundles, state: int) -> tuple:
        """ Draw a single output from the conditional distribution given state.
        """
        return self.state2bundle[state], self.y_mod.random_out(state)

    def calculate(self: Observation_with_bundles) -> numpy.ndarray:
        """Calculate and return likelihoods of states given self._y.

        Returns:
            numpy.ndarray with shape (n_times, n_states)

        Assumes self._y has been assigned by a call to self.observe().
        For each time t, the given the observation (bundle, y) return
        value is a vector of length n_states with components
        Probability(y|state)*Probability(state|bundle).

        """
        assert self.n_times is not None  # Ensure that self.observe() was called

        # Get unmasked likelihoods
        self._observed_py_state = self.y_mod.calculate()

        # Apply the right mask for the bundle at each time.  Note this
        # modifies self.y_mod._observed_py_state in place.
        for t in range(self.n_times):
            self._observed_py_state[t, :] *= self.bundle_and_state[
                self._y.bundles[t], :]

        return self._observed_py_state

    def reestimate(self: Observation_with_bundles, w: numpy.ndarray):
        """Reestimate parameters of self.y_mod

        Args:
            w: Weights with w[t,s] = alpha[t,s]*beta[t,s] = Probability(state=s|all data)

        Assumes that observations are already attached to self.y_mod by
        self.observe().

        """
        self.y_mod.reestimate(w)
