"""base.py: Versions of code in simple.py that are extendable in ways necessary for
some applications

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import typing  # For type hints

import numpy
import numpy.random

import hmm.simple


class Observation_0:
    """Base class for observations.  You can't use instances of this
    class.  You must use a subclass.

    """
    _parameter_keys = tuple([])  # Specifies parameters reported by

    # self.__str__.  These parameter
    # names should be in self.__dict__.

    def __init__(self: Observation_0, *args):
        self._rng = args[-1]
        self.n_states = 0  # Subclasses must set n_states
        for key in self._parameter_keys:
            assert key in self.__dict__

    def _concatenate(self: Observation_0, y_segs: (tuple, list)) -> tuple:
        """Concatenate observation segments.  Assumes each is a numpy array.

        Args:
            y_segs:  The observations.  The structure depends on the subclass

        Returns: (tuple): (y, t_seg) where the structure of y depends
            on the subclass, and t_seg is a list of segment
            boundaries.

        """
        assert isinstance(y_segs, (tuple, list))
        length = 0
        t_seg = [0]
        for seg in y_segs:
            length += len(seg)
            t_seg.append(length)
        try:
            all_data = numpy.concatenate(y_segs)
        except:
            print('Observation_0._concatenate() cannot handle type {0}'.format(
                type(y_segs[0])))
            raise
        return all_data, t_seg

    def reestimate(self: Observation_0, w: numpy.ndarray):
        """Based on w, modify parameters of self

        Args:
            w (numpy.ndarray):  A statistic of all the data, y, and old
                parameters of HMM including self.
                w[t, s] = Prob(state[t]=s|y and HMM)

        Sets parameters of self to \argmax_\theta Prob(y|w) (Todo:
        verify)

        """
        raise RuntimeError('Not implemented.  Use a subclass.')

    def random_out(self: Observation_0, state: int):
        """ Returns a random draw from Prob(y|state)

        Args:
            state: The state

        Returns:
            (object): A single observation

        """
        raise RuntimeError('Not implemented.  Use a subclass.')

    def calculate(self: Observation_0) -> numpy.ndarray:
        """Calculate and return the likelihoods of states given observations.

        Returns:
            self._likelihood where _likelihood[t,i] = Prob(y[t]|state[t]=i)

        """
        raise RuntimeError('Not implemented.  Use a subclass.')

    def merge(self: Observation_0, raw_outs):
        """Reformat raw_outs into suitable form for self.observe()

        Motivating example: raw_outs is a list of tuples (bundle, y)
        from Observation_with_bundles.random_out().

        """
        return numpy.array(raw_outs)

    def observe(self: Observation_0, y_segs) -> numpy.ndarray:
        """Attach observations to self

        Args:
            y_segs: Independent measurement sequences.  Structure
                specified by implementation of self._concatenate() by
                subclasses.

        Returns:
            t_segs: Segment boundaries

        """
        # pylint: disable = attribute-defined-outside-init
        self._y, t_seg = self._concatenate(y_segs)
        self.t_seg = numpy.array(t_seg)
        self.n_times = t_seg[-1]
        self._likelihood = numpy.empty((self.n_times, self.n_states))
        return self.t_seg

    def __str__(self: Observation_0) -> str:
        """Return a string representation of self.

        Returns:
            A string representation of self

        """
        return_string = 'An {0} instance:\n'.format(type(self))
        for key in self._parameter_keys:
            return_string += '    {0}\n'.format(key)
            return_string += '{0}\n'.format(getattr(self, key))
        return return_string


class IntegerObservation(Observation_0):
    r"""Observation model for integers with y[t] \in [0,y_max) \forall t

    Args:
        py_state: Conditional probability of each y give each state
        rng: A numpy.random.Generator for simulation

    """

    _parameter_keys = ('_py_state',)

    def __init__(self: IntegerObservation, py_state: numpy.ndarray,
                 rng: numpy.random.Generator):
        self._py_state = hmm.simple.Prob(py_state)
        super().__init__(rng)

        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)
        self.n_states = len(self._py_state)

    def reestimate(  # pylint: disable = arguments-differ
            self: IntegerObservation,
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
            # pylint: disable = attribute-defined-outside-init
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

    def calculate(self: IntegerObservation) -> numpy.ndarray:
        r"""
        Calculate likelihoods: self._likelihood[t,i] = P(y(t)|state(t)=i)

        Returns:
            state_likelihood[t,i] \forall t \in [0,n_times) and i \in [0,n_states)

        Assumes a previous call to measure has assigned self._y and allocated
            self._likelihood

        """

        # mypy objects ""Unsupported target for indexed assignment"
        self._likelihood[:, :] = self._py_state[:, self._y].T  # type: ignore
        return self._likelihood

    def random_out(self: IntegerObservation, state: int) -> int:
        """For simulation, draw a random observation given state s

        Args:
            state: Index of state

        Returns: Random observation drawn from distribution
            conditioned on the state

        """
        return numpy.searchsorted(self._cummulative_y[state],
                                  self._rng.random())


class HMM(hmm.simple.HMM):
    """Uses observations that are in many segments.  Also supports
    decoding sequences of bundles rather than states if the
    observation class has information about bundles.

    """

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
        # Make corrections for terms on segment boundaries
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

    def forward(  # pylint: disable = arguments-differ
            self: HMM,
            t_start: int = 0,
            t_stop: int = 0,
            t_skip: int = 0,
            last_0=None) -> float:
        """Recursively calculate state probabilities.

        Args:
            t_start: Use self.state_likelihood[t_start] first
            t_stop: Use self.state_likelihood[t_start] first
            t_skip: Number of time steps from when "last" is valid till t_start

        Returns:
            Log (base e) likelihood of HMM given entire observation sequence

        """
        if t_stop == 0:
            # Reduces to ignoring t_start and t_stop and operating on
            # a single segment
            assert t_start == 0
            t_stop = len(self.state_likelihood)

        if last_0 is None:
            last_0 = self.p_state_initial

        last = numpy.copy(last_0).reshape(-1)
        for t in range(t_skip):
            self.p_state2state.step_forward(last)

        for t in range(t_start, t_stop):
            last *= self.state_likelihood[t]  # Element-wise multiply
            assert last.sum() > 0
            self.gamma_inv[t] = 1 / last.sum()
            last *= self.gamma_inv[t]
            self.alpha[t, :] = last
            self.p_state2state.step_forward(last)
        return -(numpy.log(self.gamma_inv[t_start:t_stop])).sum()

    def backward(  # pylint: disable = arguments-differ
            self: HMM, t_start=0, t_stop=0):
        """
        Baum Welch backwards pass through state conditional likelihoods.


        Calculates values of self.beta which "reestimate()" needs.
        """

        if t_stop == 0:
            # Reduces to ignoring t_start and t_stop and operating on
            # a single segment
            assert t_start == 0
            t_stop = len(self.state_likelihood)

        last = numpy.ones(self.n_states)
        for t in range(t_stop - 1, t_start - 1, -1):
            self.beta[t, :] = last
            last *= self.state_likelihood[t] * self.gamma_inv[t]
            self.p_state2state.step_back(last)

    def multi_train(self: HMM, ys, n_iterations: int, display=True):
        """Train on more than one independent sequence of observations

        Args:
            ys: Measured observation sequences in format appropriate
                for self.y_mod
            n_iter: The number of iterations to execute

        Returns:
            List of log likelihood per observation for each iteration

        The differences from base.HMM: 1. More than one independent
        observation sequence.  2. The structure of observations is not
        specified.

        For the first training iteration, the initial distribution of
        states at the beginning of each observation sequence is given
        by self.p_state_initial.  However, each training iteration
        updates the distribution of the initial state for each
        observation sequence independently.  At the end of training,
        the average of those independent distributions for the initial
        states is assigned to self.p_state_initial.

        I don't know of assumptions for which that procedure is
        exactly correct.  If one knows that each observation segment
        starts with the same state distribution, then one should use a
        single initial state distribution in training.  If one knows
        that the initial state distributions are different and that
        when the model is used it will be applied to new data drawn
        from similar sets of segments, then one should train with
        separate initial state distributions and retain them for use
        in the future.

        The compromise treatment of the initial distribution of states
        here is intended to treat initial distributions that are
        different from each other but more similar to each other than
        the time averaged state distribution.

        """

        # pylint: disable = attribute-defined-outside-init

        # log_like_list[i] = log(Prob(ys|HMM[iteration=i]))/n_times, ie,
        # the log likelihood per time step
        log_like_list = []

        t_seg = self.y_mod.observe(ys)  # Segment boundaries
        self.n_times = self.y_mod.n_times

        assert self.n_times > 1
        assert t_seg[0] == 0

        n_seg = len(t_seg) - 1
        self.alpha = numpy.empty((self.n_times, self.n_states))
        self.beta = numpy.empty((self.n_times, self.n_states))
        self.gamma_inv = numpy.empty((self.n_times,))

        # State probabilities at the beginning of each segment
        p_state_initial_all = numpy.empty((n_seg, self.n_states))
        for seg in range(n_seg):
            p_state_initial_all[seg, :] = self.p_state_initial.copy()

        for iteration in range(n_iterations):
            message = "i={0:4d} ".format(iteration)
            sum_log_like = 0.0
            self.state_likelihood = self.y_mod.calculate()

            # Operate on each observation segment separately and put
            # the results in the corresponding segement of the alpha,
            # beta and gamma arrays.

            for seg in range(n_seg):
                # Set up self to run forward and backward on this segment
                t_start = t_seg[seg]
                t_stop = t_seg[seg + 1]

                log_likelihood = self.forward(
                    t_start, t_stop, last_0=p_state_initial_all[seg, :])
                self.backward(t_start, t_stop)

                sum_log_like += log_likelihood
                p_state_initial_all[
                    seg, :] = self.alpha[t_start] * self.beta[t_start]
                # Flag to prevent fitting state transitions between segments
                self.gamma_inv[t_start] = -1
                message += "L[{0}]={1:7.4f} ".format(
                    seg, log_likelihood / self.n_times)

            self.reestimate()

            # Record/report/check this iteration
            log_like_list.append(sum_log_like / self.n_times)
            message += "avg={0:10.7f}".format(log_like_list[-1])
            self.ensure_monotonic(log_like_list, display, message)

        self.p_state_initial[:] = p_state_initial_all.sum(axis=0)
        self.p_state_initial /= self.p_state_initial.sum()
        return log_like_list

    def initialize_y_model(
            self: HMM,
            y,
            state_sequence: typing.Optional[numpy.ndarray] = None):
        """Given data, make plausible y_model.

        Args:
            y: Observation sequence.  Type must work for self.y_mod.observe(y)
            state_sequence: Optional state sequence

        Use this method to create an observation model that is not the
        same for every state and that makes plausible the data in the
        argument y plausible.

        """
        # ToDo: Fails for general y_mod
        n_times = self.y_mod.observe(y)[-1]
        if state_sequence is None:
            state_sequence = numpy.array(self.state_simulate(n_times),
                                         numpy.int32)

        # Set alpha to enforce the simulated state sequence in reestimate
        alpha = numpy.zeros((n_times, self.n_states))
        for t in range(n_times):
            alpha[t, state_sequence[t]] = 1

        self.y_mod.reestimate(alpha)
        return self.y_mod

    def simulate(self: HMM, length: int):
        """Simulate n steps of HMM

        Args:
            length: Number of time steps to simulate

        Returns:
            (states, outs) where states is a state sequence and outs
            is a sequence of observations

        """

        states, raw_outs = super().simulate(length)
        # y_mod.merge enables funny observation models like
        # Observation_with_bundles
        return states, self.y_mod.merge(raw_outs)

    def broken_decode(self: HMM, y) -> numpy.ndarray:
        """Implements the flawed algorithm from first edition of the book.

        Args:
            self: An HMM with an observation model for Bundle_segment instances
            y[0]: A single observation sequence appropriate for the underlying
                observation model

        Returns:
           Flawed estimate of the maximum likelihood sequence of bundles
        """
        assert len(y) == 1

        bundle2state_dict = self.y_mod.bundle2state  # Dict
        n_bundles = self.y_mod.n_bundle
        bundle_and_state = self.y_mod.bundle_and_state

        underlying = self.y_mod.underlying_model

        t_seg = underlying.observe(y)
        n_times = t_seg[-1]
        likelihood = underlying.calculate()

        # predecessor[t, i_bundle] is the bundle[t-1] in the best
        # sequence bundle[:t] given y[:t] and bundle[t] = i_bundle
        predecessor = numpy.empty((n_times, n_bundles), dtype=numpy.int32)

        # Prepare to start iterating at t=1

        # At time t, score_state[i_state] = score of i_state given
        # previous ys and selected predecessor bundles
        score_state = self.p_state_time_average * likelihood[0]

        # At time t, score_bundle[i_bundle] = score of i_bundle given
        # previous ys and selected predecessor bundles
        score_bundle = numpy.dot(bundle_and_state, score_state)

        # Make the score of each state in a bundle = the score of that
        # state given the bundle todo ?
        for bundle in range(n_bundles):
            if score_bundle[bundle] == 0.0:
                continue
            score_state[bundle2state_dict[bundle]] /= score_bundle[bundle]

        for t in range(1, n_times):
            # Score of sequential pairs of states (todo: wrong)
            state2state_score = self.p_state2state.cost(
                numpy.dot(score_bundle, bundle_and_state) * score_state,
                likelihood[t])

            # Score of bundle@t and state@(t-1)
            bundle2state_score = numpy.dot(bundle_and_state, state2state_score)

            # Score of a sequential pair of bundles
            bundle2bundle_score = numpy.dot(bundle2state_score,
                                            bundle_and_state.T)

            # Find best predecessor bundle
            predecessor[t] = bundle2bundle_score.argmax(axis=0)

            # Assign score_bundle given chosen bundle
            score_bundle = numpy.choose(predecessor[t], bundle2bundle_score)
            divisor = score_bundle.max()
            if divisor == 0.0:
                print('Stuck with an impossible bundle sequence.  Hacking now')
                score_bundle[:] = 1.0
                score_state[:] = 1.0
            score_bundle /= score_bundle.max()

            # Forget state history (todo: wrong?) and assign
            # score_state in each bundle conditional on selected
            # predecessor bundle
            for bundle in range(n_bundles):
                states = bundle2state_dict[bundle]
                old_bundle = predecessor[t, bundle]
                score_state[states] = bundle2state_score[old_bundle, states]
                divisior = score_state[states].sum()
                if divisior == 0.0:
                    continue
                score_state[states] /= divisior
        # Backtrack
        last_bundle = numpy.argmax(score_bundle)
        sequence = numpy.empty((n_times,), dtype=numpy.int32)
        for t in range(n_times - 1, -1, -1):
            sequence[t] = last_bundle
            last_bundle = predecessor[t, last_bundle]
        return sequence


class Bundle_segment:
    """A pair of time series: bundle ids and observations.

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
        return """y values:{0:s}
bundle values:{1:s}
""".format(self.y, self.bundles)


class Observation_with_bundles(Observation_0):
    """Represents likelihood of states given observations by masking an
    underlying likelihood model.

    Args:
        underlying_instance: An instance of the underlying model class
        bundle2state: Keys are bundle ids and values are lists of states
        rng: A numpy.random.Generator for simulation
        small: Threshold for plausible sum of state likelihoods at a time

    Attributes:

    underlying_model:  Provides methods for observations without bundle tags
    bundle2state: A dict with bundle2state[i_bundle] = list of states
    n_bundle:
    n_states:
    bundle_and_state: A boolean array with state in bundle <=>
         bundle_and_state[i_bundle, i_state] = True
    state2bundle: i_bundle = state2bundle[i_state] with state in bundle
    """
    _parameter_keys = 'underlying_model bundle2state'.split()

    def __init__(self: Observation_with_bundles,
                 underlying_instance: Observation_0,
                 bundle2state: dict,
                 rng: numpy.random.Generator,
                 small=1.0e-30):
        self.underlying_model = underlying_instance
        assert isinstance(self.underlying_model, Observation_0)
        self.bundle2state = bundle2state
        self.n_bundle = len(self.bundle2state)
        self.small = small
        # Call super().__init__ to check all _parameter_keys assigned
        super().__init__(rng)

        # Find out how many states there are and ensure that each
        # state is in only one bundle.
        states = set()
        for x in self.bundle2state.values():
            # Assert that no state is in more than one bundle.  Think
            # before relaxing this.
            assert states & set(x) == set()
            states = states | set(x)
        self.n_states = len(states)
        # Ensure that states is a set of sequential integers [0,n_states)
        assert states == set(range(self.n_states))

        # bundle_and_state[bundle_id, state_id] is true iff state \in bundle.
        self.bundle_and_state = numpy.zeros((self.n_bundle, self.n_states),
                                            numpy.bool)

        # state2bundle[state_id] = bundle_id for bundle that contains state
        self.state2bundle = numpy.ones(self.n_states, dtype=numpy.int32) * -1

        for bundle_id, states in self.bundle2state.items():
            for state_id in states:
                self.bundle_and_state[bundle_id, state_id] = True
                # Ensure that no state is in more than one bundle
                assert self.state2bundle[state_id] == -1
                self.state2bundle[state_id] = bundle_id

    def observe(  # pylint: disable = arguments-differ
            self: Observation_with_bundles, bundle_segment_list: list) -> int:
        """Attach observations to self as a single Bundle_segment

        Args:
            bundle_segment_list: List of Bundle_segments

        Side effects: 1. Assign self.t_seg; 2. Call self.y_mod.observe
        to attach observations stripped of bundle tags to self.y_mod.

        """
        # pylint: disable = attribute-defined-outside-init
        self._y, self.t_seg = self._concatenate(bundle_segment_list)
        self.n_times = self.t_seg[-1]
        self.underlying_model.observe([self._y.y])
        return self.t_seg

    def _concatenate(  # pylint: disable = arguments-differ
            self: Observation_with_bundles,
            bundle_segment_list: list) -> Bundle_segment:
        """ Create a single Bundle_segment from a list of segments

        Args:
            bundle_segment_list: Each element is a Bundle_segment

        Returns:
            (y, t_seg)
        """
        length = 0
        t_seg = [0]
        bundles = []
        ys = []
        for segment in bundle_segment_list:
            length += len(segment.bundles)
            t_seg.append(length)
            bundles.append(segment.bundles)
            ys.append(segment.y)

        all_bundles = [item for sublist in bundles for item in sublist]
        all_ys, y_t_seg = self.underlying_model._concatenate(ys)
        assert tuple(t_seg) == tuple(y_t_seg)
        return Bundle_segment(all_bundles, all_ys), t_seg

    def random_out(self: Observation_with_bundles, state: int) -> tuple:
        """ Draw a single output from the conditional distribution given state.
        """
        return self.state2bundle[state], self.underlying_model.random_out(state)

    def merge(self: Observation_with_bundles,
              raw_outs: list) -> Observation_with_bundles:
        """ Merge isolated pairs (bundle, y) into an Observation_with_bundles.

        Args:
            raw_outs: A list of pairs (bundle[t], y[t])

        Returns:
            A single Observation_with_bundles instance

        """
        bundles = []
        y = []
        for out in raw_outs:
            bundles.append(out[0])
            y.append(out[1])

        return Bundle_segment(numpy.array(bundles),
                              self.underlying_model.merge(y))

    def calculate(self: Observation_with_bundles) -> numpy.ndarray:
        """Calculate and return likelihoods of states given self._y.

        Returns:
            likelihood with likelihood[t,s] = Probability(y[t]|state[t]=s) *
                Probability(state=s|bundle[t])

        Assumes self._y has been assigned by a call to self.observe().
        For each time t, the given the observation (bundle, y) return
        value is a vector of length n_states with components
        Probability(y|state)*Probability(state|bundle).

        """

        # pylint: disable = attribute-defined-outside-init
        # Get unmasked likelihoods
        self._likelihood = self.underlying_model.calculate()

        # Apply the right mask for the bundle at each time.  Note this
        # modifies self.y_mod._likelihood in place.
        for t in range(self.n_times):
            self._likelihood[t, :] *= self.bundle_and_state[
                self._y.bundles[t], :]
            if self._likelihood[t, :].sum() < self.small:
                raise ValueError(
                    'Observation is not plausible from any state.  ' +
                    'self.likelihood[{0},:]=\n{1}'.format(
                        t, self._likelihood[t, :]))

        return self._likelihood

    def reestimate(self: Observation_with_bundles, w: numpy.ndarray):
        """Reestimate parameters of self.y_mod

        Args:
            w: Weights with w[t,s] = alpha[t,s]*beta[t,s] =
                Probability(state=s|all data)

        Assumes that observations are already attached to self.y_mod by
        self.observe().

        """
        self.underlying_model.reestimate(w)
