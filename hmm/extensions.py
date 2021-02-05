""" extensions.py: Extensions of base.py that are necessary for some applications

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import typing  # For type hints

import numpy
import numpy.random

import hmm.base


class Observation(hmm.base.Observation):
    """Ancestor of other observation classes

    Args:
        parameters: A dict {'p_ys': array}. Use dict for flexible sub-classes
        rng: A numpy.random.Generator for simulation

    Differences from hmm.base.Observation: 1. Observed data
    is a list of sequences; 2. Parameters are passed as a dict to
    support subclasses.

    Public methods and attributes:

    __init__

    observe

    random_out

    calculate

    reestimate

    t_seg         Assigned by self.observe and used in multi_train

    _py_state because it's name is in an argument that's a dict

    """
    _parameter_keys = set(('_py_state',))

    def __init__(  # pylint: disable = super-init-not-called
            self: Observation, parameters: dict, rng: numpy.random.Generator):
        assert set(parameters.keys()) == self._parameter_keys
        for key, value in parameters.items():
            setattr(self, key, value)
        self._rng = rng
        self.n_states = self._normalize()
        self._likelihood = None
        self.n_times = None  # Flag to be set to an int by self.observe()

    def __str__(self: Observation) -> str:
        return_string = 'An {0} instance:\n'.format(type(self))
        for key in self._parameter_keys:
            return_string += '    {0}\n'.format(key)
            return_string += '{0}\n'.format(getattr(self, key))
        return return_string

    def observe(  # pylint: disable = arguments-differ
            self: Observation,
            y_segs: tuple,
            n_times: typing.Optional[int] = None) -> int:
        """ Attach measurement sequence[s] to self.

        Args:
            y_segs: Any number of independent measurement segments

        Returns:
            Length of observation sequence
        """
        self._y = self._concatenate(y_segs)
        t_seg = [0]  # List of segment boundaries in concatenated ys
        length = 0
        for seg in y_segs:
            length += len(seg)
            t_seg.append(length)
        self.t_seg = numpy.array(t_seg)
        assert self.t_seg[-1] == len(self._y)
        self._likelihood = numpy.empty((len(self._y), self.n_states))
        self.n_times = len(self._y)
        if n_times:
            assert n_times == self.n_times
        return self.n_times

    def _concatenate(self: Observation, y_segs: tuple):
        """Concatenate observation segments each of which is a numpy array.

        """
        assert isinstance(y_segs, (tuple, list))
        if len(y_segs) == 1:
            return y_segs[0]
        assert len(y_segs) > 1
        # ToDo: Test this concatenation
        return numpy.concatenate(y_segs)

    def reestimate(self: Observation,
                   w: numpy.ndarray,
                   warn: typing.Optional[bool] = True):
        """
        Estimate new model parameters

        Args:
            w: w[t,s] = Prob(state[t]=s) given data and
                 old model
            warn: If True and y[0].dtype != numpy.int32, print
                warning
        """
        if not (isinstance(self._y, numpy.ndarray) and
                (self._y.dtype == numpy.int32)):
            self._y = numpy.array(self._y, numpy.int32)
            if warn:
                print("Warning: reformatted y in reestimate")
        assert self._y.dtype == numpy.int32 and self._y.shape == (
            self.n_times,), """
                y.dtype=%s, y.shape=%s""" % (
                self._y.dtype,
                self._y.shape,
            )
        for yi in range(self._py_state.shape[1]):
            self._py_state.assign_col(
                yi,
                w.take(numpy.where(self._y == yi)[0], axis=0).sum(axis=0))
        self._py_state.normalize()
        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)


class HMM(hmm.base.HMM):

    def reestimate(self: HMM):
        """Phase of Baum Welch training that reestimates model parameters

        Using values af self.alpha and self.beta calculated by
        forward() and backward(), this code updates state transition
        probabilities and initial state probabilities.  The call to
        y_mod.reestimate() updates observation model parameters.

        """

        # u_sum[i,j] = \sum_{t:gamma_inv[t+1]>0} alpha[t,i] * beta[t+1,j] *
        # state_likelihood[t+1,j]/gamma[t+1]
        #
        # The term at t is the conditional probability that there was
        # a transition from state i to state j at time t given all of
        # the observed data
        u_sum = numpy.einsum(
            "ti,tj,tj,t->ij",  # Specifies the i,j indices and sum over t
            self.alpha[:-1],  # indices t,i
            self.beta[1:],  # indices t,j
            self.state_likelihood[1:],  # indices t,j
            self.gamma_inv[1:]  # index t
        )
        # Correct for terms on segment boundaries
        for t in numpy.nonzero(self.gamma_inv < 0)[0]:
            assert self.gamma_inv[t] < 0
            if t == 0:
                continue
            for i in range(self.n_states):
                for j in range(self.n_states):
                    u_sum[i, j] -= self.alpha[t - 1, i] * self.beta[
                        t, j] * self.state_likelihood[t, j] * self.gamma_inv[t]

        self.alpha *= self.beta  # Saves allocating a new array for
        alpha_beta = self.alpha  # the result

        self.p_state_time_average = alpha_beta.sum(axis=0)
        self.p_state_initial = numpy.copy(alpha_beta[0])
        for x in (self.p_state_time_average, self.p_state_initial):
            x /= x.sum()
        assert u_sum.shape == self.p_state2state.shape
        self.p_state2state *= u_sum
        self.p_state2state.normalize()
        self.y_mod.reestimate(alpha_beta)

    def multi_train(self: HMM, ys, n_iterations: int, display=True):
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
        for iteration in range(n_iterations):
            if display:
                print("i=%d: " % iteration, end="")
            log_like_iteration = 0.0
            # Both forward() and backward() should operate on each
            # training segment and put the results in the
            # corresponding segement of the the alpha, beta and gamma
            # arrays.
            likelihood_all = self.y_mod.calculate()
            for seg in range(n_seg):
                # Set up self to run forward/backward on a segment of y
                self.n_times = t_seg[seg + 1] - t_seg[seg]
                self.alpha = alpha_all[t_seg[seg]:t_seg[seg + 1], :]
                self.beta = beta_all[t_seg[seg]:t_seg[seg + 1], :]
                self.state_likelihood = likelihood_all[t_seg[seg]:t_seg[seg +
                                                                        1]]
                self.gamma_inv = gamma_inv_all[t_seg[seg]:t_seg[seg + 1]]
                self.p_state_initial = p_state_initial_all[seg, :]

                # Run forward/backward to calculate alpha and beta
                # values for a segment of y
                log_like = self.forward()  # Log Likelihood
                if display:
                    print("L[%d]=%7.4f " % (seg, log_like / self.n_times),
                          end="")
                log_like_iteration += log_like
                self.backward()
                p_state_initial_all[seg, :] = self.alpha[0] * self.beta[0]
                # Don't fit state transitions between segments
                self.gamma_inv[0] = -1

            log_like_list.append(log_like_iteration / self.n_times)
            if iteration > 0:
                ll = log_like_list[iteration]
                ll_prev = log_like_list[iteration - 1]
                delta = ll - ll_prev
                if delta / abs(ll) < -1.0e-14:  # Todo: Why not zero?
                    print("""
WARNING training is not monotonic: avg[{0}]={1} and avg[{2}]={3} difference={4}
""".format(iteration - 1, ll_prev, iteration, ll, delta))
            if display:
                print("avg={0:10.7}f".format(log_like_list[-1]))

            # Associate all of the alpha and beta segments with self
            # and reestimate()
            self.alpha = alpha_all
            self.beta = beta_all
            self.gamma_inv = gamma_inv_all
            self.state_likelihood = likelihood_all
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


class Observation_with_bundles(Observation):
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
        self._likelihood = self.y_mod.calculate()

        # Apply the right mask for the bundle at each time.  Note this
        # modifies self.y_mod._likelihood in place.
        for t in range(self.n_times):
            self._likelihood[t, :] *= self.bundle_and_state[
                self._y.bundles[t], :]

        return self._likelihood

    def reestimate(self: Observation_with_bundles, w: numpy.ndarray):
        """Reestimate parameters of self.y_mod

        Args:
            w: Weights with w[t,s] = alpha[t,s]*beta[t,s] = Probability(state=s|all data)

        Assumes that observations are already attached to self.y_mod by
        self.observe().

        """
        self.y_mod.reestimate(w)
