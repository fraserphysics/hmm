""" base.py: Implements basic HMM algorithms.

Classes:

    :py:class:`Observation`:
        Models of discrete observations

    :py:class:`HMM`:
        A Hidden Markov Model implementation
"""
# Nomenclature:
#
# y:                Observations
#
# y_mod:            An instance of an observation model, eg, Observation in this file
#
# n_times:          The number of time points in data for an observable
#
# state_likelihood: Given observed data y[t] = y_ and states[t] = s_,
#                   state_likelihood[t, s_] = Prob(y_ | s_)
#

# pylint: disable = attribute-defined-outside-init
from __future__ import annotations  # Enables, eg, (self: HMM,

import typing  # For type hints

import numpy
import numpy.random

COPYRIGHT = """Copyright (c) 2021 Andrew M. Fraser

This file is part of hmm.

Hmm is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Hmm is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

See the file gpl.txt in the root directory of the hmm distribution
or see <http://www.gnu.org/licenses/>.
"""


class HMM:
    """A Hidden Markov Model implementation.

    Args:
        p_state_initial : Initial distribution of states
        p_state_time_average : Stationary distribution of states
        p_state2state : Probability of state given state:
            p_state2state[a, b] = Prob(s(1)=b|s(0)=a)
        y_mod : Instance of class for probabilities of observations
        rng : Numpy generator with state

    p_state_time_average is averaged over training data.  It is not
    the stationary distribution of p_state2state.

    Arguments are passed by reference and they are modified by some of
    the methods of HMM.

    By initializing with rng created by the caller with, eg
    numpy.random.default_rng(), one can ensure reproducible
    pseudo-random sequences and avoid using the same pseudo-random
    sequences in different parts of the code.

    """

    # p_state_time_average, p_state_transition, observation_model
    def __init__(self: HMM,
                 p_state_initial: numpy.ndarray,
                 p_state_time_average: numpy.ndarray,
                 p_state2state: numpy.ndarray,
                 y_mod: Observation,
                 rng: typing.Optional[numpy.random.Generator] = None) -> None:

        if rng is None:
            self.rng = numpy.random.default_rng()
        else:
            self.rng = rng
        self.n_states = len(p_state_initial)
        self.p_state_initial = numpy.array(p_state_initial)
        self.p_state_time_average = numpy.array(p_state_time_average)
        self.p_state2state = Prob(numpy.array(p_state2state))
        self.y_mod = y_mod

    # Todo: Perhaps handle short bursts of missing data, ie, sequences
    # are not independent.

    def forward(self: HMM) -> float:
        """Recursively calculate state probabilities.

        Returns:
            Average log (base e) likelihood per point of entire observation sequence

        Requires that observation probabilities have already been calculated

        On entry:

        - self                    is an HMM

        - self.state_likelihood   has been calculated

        - self.alpha              has been allocated

        - self.gamma_inv          has been allocated

        On return:

        - 1/self.gamma_inv[t] = Prob{y(t)=y(t)|y_0^{t-1}}
        - self.alpha[t, i] = Prob{s(t)=i|y_0^t}

        """

        # last is a conditional distribution of state probabilities.
        # What it is conditioned on changes as the calculations
        # progress.
        last = numpy.copy(self.p_state_initial.reshape(-1))  # Copy
        for t in range(len(self.state_likelihood)):
            last *= self.state_likelihood[t]  # Element-wise multiply
            assert last.sum() > 0
            self.gamma_inv[t] = 1 / last.sum()
            last *= self.gamma_inv[t]
            self.alpha[t, :] = last
            # Could use Prob.step_forwar()
            last[:] = numpy.dot(last, self.p_state2state)
        return -(numpy.log(self.gamma_inv)).sum()

    def backward(self: HMM) -> None:
        """
        Baum Welch backwards pass through state conditional likelihoods.


        Calculates values of self.beta which "reestimate()" needs.

        On entry :

        - self               is an HMM

        - self.state_likelihood  has been calculated

        - self.gamma_inv     has been calculated by forward

        - self.beta          has been allocated

        On return:

        - For each state i, beta[t, i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}

        """
        # last and beta are analogous to last in alpha in forward(),
        # but the precise interpretations are more complicated.
        last = numpy.ones(self.n_states)
        for t in range(len(self.state_likelihood) - 1, -1, -1):
            self.beta[t, :] = last
            last *= self.state_likelihood[t]
            last *= self.gamma_inv[t]
            # Could use Prob.step_back()
            last[:] = numpy.dot(self.p_state2state, last)

    def train(
            self: HMM,
            y,  #  Type must work for self.y_mod.observe(y)
            n_iterations: int = 1,
            display: typing.Optional[bool] = True) -> list:
        """Use Baum-Welch algorithm to search for maximum likelihood
        model parameters.

        Args:
            y: Measurements appropriate for self.y_mod
            n_iter: The number of iterations to execute
            display: If True, print the log likelihood
                per observation for each iteration

        Returns:
            List of log likelihood per observation for each iteration

        """

        log_likelihood_list = []
        # Attach observations to self.y_mod
        self.y_mod.observe(y)
        self.n_times = self.y_mod.n_times
        assert self.n_times > 1

        # Allocate working arrays
        self.alpha = numpy.empty((self.n_times, self.n_states))
        self.beta = numpy.empty((self.n_times, self.n_states))
        self.gamma_inv = numpy.empty((self.n_times,))

        for iteration in range(n_iterations):
            self.state_likelihood = self.y_mod.calculate()
            log_likelihood = self.forward()
            self.backward()
            self.reestimate()

            log_likelihood_list.append(log_likelihood / self.n_times)
            self.ensure_monotonic(
                log_likelihood_list, display,
                "{0:4d}: LLps={1:7.3f}".format(iteration,
                                               log_likelihood_list[-1]))

        return log_likelihood_list

    def ensure_monotonic(self: HMM, log_likelihood_list, display, message):
        if display:
            print(message)
        if len(log_likelihood_list) == 1:
            return

        ll = log_likelihood_list[-1]
        ll_prev = log_likelihood_list[-2]
        delta = ll - ll_prev
        if delta / abs(ll) < -1.0e-14:  # Todo: Why not zero?
            iteration = len(log_likelihood_list)
            raise ValueError("""
WARNING training is not monotonic: LLps[{0}]={1} and LLps[{2}]={3} difference={4}
""".format(iteration - 1, ll_prev, iteration, ll, delta))

    def reestimate(self: HMM):
        """Phase of Baum Welch training that reestimates model parameters

        Using values af self.alpha and self.beta calculated by
        forward() and backward(), this code updates state transition
        probabilities and initial state probabilities.  The call to
        y_mod.reestimate() updates observation model parameters.

        """

        # u_sum[i,j] = \sum_t alpha[t,i] * beta[t+1,j] *
        # state_likelihood[t+1]/gamma[t+1]
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

    def decode(self: HMM, y) -> numpy.ndarray:
        """
        Find the most likely state sequence for given observation sequence.

        Args:
            y: Observations with type for self.y_mod or None if
                self.state_likelihood was assigned externally.

        Returns:
            Maximum likelihood state sequence

        This implements the Viterbi algorithm.
        """

        if y is None:
            print("""Warning: No y argument to decode().  Assuming
self.state_likelihood was assigned externally.""")
        else:  # Calculate likelihood of data given state
            self.y_mod.observe(y)
            self.n_times = self.y_mod.n_times
            self.state_likelihood = self.y_mod.calculate()
        n_times, n_states = self.state_likelihood.shape
        assert self.n_states == n_states
        assert n_times > 1

        # Allocate working memory
        best_predecessors = numpy.empty((self.n_times, self.n_states),
                                        numpy.int32)
        best_state_sequence = numpy.ones((self.n_times, 1), numpy.int32)

        # Use initial state distribution for first best_path_cost
        best_path_cost = self.state_likelihood[0] * self.p_state_initial

        for t in range(1, self.n_times):
            # cost = p_state2state*outer(best_path_cost, state_likelihood[t])
            # Could use Prob.cost()
            cost = (self.p_state2state.T *
                    best_path_cost).T * self.state_likelihood[t]
            best_predecessors[t] = cost.argmax(axis=0)
            best_path_cost = numpy.choose(best_predecessors[t], cost)
            if best_path_cost.max() == 0:
                raise ValueError(
                    "Attempt to decode impossible observation sequence")
            best_path_cost /= best_path_cost.max()  # Prevent underflow

        # Find the best end state
        previous_best_state = numpy.argmax(best_path_cost)

        # Backtrack through best_predecessors to find the best
        # sequence.
        for t in range(self.n_times - 1, -1, -1):
            best_state_sequence[t] = previous_best_state
            previous_best_state = best_predecessors[t, previous_best_state]
        return best_state_sequence.flat

    def initialize_y_model(
        self: HMM,
        y,  #  Type must work for self.y_mod.observe(y)
        state_sequence: typing.Optional[numpy.ndarray] = None):
        """ Given data, make plausible y_model.

        Args:
            y: Observation sequence
            state_sequence: State sequence

        """
        n_times = self.y_mod.observe(y)
        if state_sequence is None:
            state_sequence = numpy.array(self.state_simulate(n_times),
                                         numpy.int32)

        # Set alpha and beta so that in reestimate they enforce the
        # simulated state sequence
        self.alpha = numpy.zeros((n_times, self.n_states))
        t = numpy.arange(n_times)
        self.alpha[t, state_sequence] = 1
        self.beta = self.alpha

        self.gamma_inv = numpy.ones(n_times)
        self.state_likelihood = numpy.ones((n_times, self.n_states))
        self.reestimate()
        return self.y_mod

    def state_simulate(
        self: HMM,
        length: int,
        mask: typing.Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        """Generate a random sequence of states that is perhaps constrained
        by a mask.

        Args:
            length: Length of returned array

        Keyword Args:
            mask: If mask[t, i] is False, state i is forbidden at time t.

        Returns:
            Sequence of states

        The returned sequence is not a draw from the random process
        defined by the model.  However the sequence has probability >
        0.

        """

        self.state_likelihood = self.rng.random((length, self.n_states))
        self.n_times = length
        if mask is not None:
            self.state_likelihood *= mask

        try:
            state_sequence = self.decode(None)
        except (ValueError):
            raise ValueError(
                "State_simulate given an impossible mask constraint")

        return state_sequence

    def simulate(
        self: HMM,
        length: int,
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Generate a random sequence of observations of a given length.

        Args:
            length: Number of time steps to simulate

        Returns:
            (states, outs) where states[t] is the state at time t, and
                outs[t] is the output at time t.

        """

        # Initialize lists
        outs = []
        states = []
        # Set up cumulative distributions
        cumulative_initial = numpy.cumsum(self.p_state_time_average[0])
        cumulative_transition = numpy.cumsum(self.p_state2state.values(),
                                             axis=1)

        # cum_rand generates random integers from a cumulative distribution
        def cum_rand(cum):
            return numpy.searchsorted(cum, self.rng.random())

        # Select an initial state
        state = cum_rand(cumulative_initial)
        # Select subsequent states and call model to generate observations
        for _ in range(length):
            states.append(state)
            outs.append(self.y_mod.random_out(state))
            state = cum_rand(cumulative_transition[state])
        return states, outs

    def link(self: HMM, here: int, there: int, p: float):
        """Create (or remove) a link between state "here" and state "there".

        Args:
            here: One index of element to modify
            there: Another index of element to modify
            p: Weight or probability of link

        The strength of the link is a function of both the argument
        "p" and the existing conditional probabilities of state
        transitions, self.p_state2state, in which
        self.p_state2state[here, there] is the probability of going to
        state there given that the system is in state here.  The code
        sets p_state2state[here, there] to p and then re-normalizes.
        Set self.p_state2state itself if you need to set exact values.
        You can use this method to modify the state topology before
        training.

        """
        self.p_state2state[here, there] = p
        self.p_state2state[here, :] /= self.p_state2state[here, :].sum()

    def __str__(self: HMM) -> str:  # HMM instance
        #save = numpy.get_printoptions
        # numpy.set_printoptions(precision=3)
        rv = """{0} with {1:d} states
p_state_initial:      {2}
p_state_time_average: {3}
p_state2state =
{4}
{5}""".format(
            self.__class__,
            self.n_states,
            self.p_state_initial,
            self.p_state_time_average,
            self.p_state2state.values(),
            self.y_mod,
        )
        # numpy.set_printoptions(save)
        return rv

    def deallocate(self: HMM):
        """ Remove arrays assigned by train.

        To be called before writing a model to disk
        """
        del (self.alpha)
        del (self.beta)
        del (self.gamma_inv)


class Observation:
    """ Probability models for observations drawn from a set of sequential integers.

    Args:
        py_state:  Conditional probability of y given state
        rng: A numpy.random.Generator for simulation

    Public methods and attributes:

    __init__

    observe

    random_out

    calculate

    reestimate

    _py_state  # Todo: rename to _py_state and make it private

    """

    def __init__(self: Observation,
                 py_state: numpy.ndarray,
                 rng: numpy.random.Generator = None):
        self._py_state = py_state
        if rng is None:
            self._rng = numpy.random.default_rng()
        else:
            self._rng = rng
        self.n_states = self._normalize()
        # self._likelihood[t,s] = Prob(y[t]|state[t]=s).  Assigned in
        # self.calculate().
        self._likelihood = None

    def _normalize(self: Observation) -> int:
        """ Separate from __init__ to make subclass easy

        Returns:
           (int): Number of states
        """
        self._py_state = Prob(self._py_state)
        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)
        return len(self._py_state)  # n_states

    def observe(self: Observation, y) -> int:
        """ Attach measurement sequence[s] to self.

        Args:
            y: A sequence of integer observations

        Returns:
            Length of observation sequence
        """
        self._y = y
        self.n_times = len(self._y)

        # Allocate here rather than in calculate() because calculate()
        # may be called more often than observe().
        self._likelihood = numpy.empty((self.n_times, self.n_states),
                                       dtype=numpy.float64)
        return self.n_times

    def calculate(self: Observation) -> numpy.ndarray:
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

    def random_out(self: Observation, state: int) -> int:
        """For simulation, draw a random observation given state s

        Args:
            state: Index of state

        Returns: Random observation drawn from distribution
            conditioned on the state

        """
        return numpy.searchsorted(self._cummulative_y[state],
                                  self._rng.random())

    def reestimate(
        self: Observation,
        w: numpy.ndarray,
        warn: typing.Optional[bool] = True,
    ):
        """
        Estimate new _py_state

        Args:
            w: w[t,s] = Prob(state[t]=s) given data and
                 old model
            warn: If True and y[0].dtype != numpy.int32, print
                warning
        """

        # Todo: Move concerns about dtype to subclasses
        if not (isinstance(self._y, numpy.ndarray) and
                (self._y.dtype == numpy.int32)):
            self._y = numpy.array(self._y, numpy.int32)
            if warn:
                print("Warning: reformatted y in reestimate")

        assert self._y.dtype == numpy.int32 and self._y.shape == (self.n_times,)

        # Loop over range of allowed values of y
        for yi in range(self._py_state.shape[1]):
            # yi was observed at times: numpy.where(self._y == yi)[0]
            # w.take(...) is the conditional state probabilities at those times
            self._py_state.assign_col(
                yi,
                w.take(numpy.where(self._y == yi)[0], axis=0).sum(axis=0))
        self._py_state.normalize()
        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)


class Prob(numpy.ndarray):
    """Subclass of ndarray for probability matrices.  P[a,b] is the
    probability of b given a.  The class has additional methods and is
    designed to enable alternative implementations that run faster or
    in less memory but may be implemented by uglier code.

    """

    def __new__(cls, x: numpy.ndarray):
        """ Return a Prob instance of the argument.

        Args:
            x: An array of probabilities

        """
        assert len(x.shape) == 2
        # cls is Prob.  This calls __new__ of numpy.ndarray and makes
        # the return value a Prob instance.
        return super().__new__(cls, x.shape, buffer=x.data)

    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def normalize(self: Prob) -> Prob:  # Prob instance
        """
        Make each row sum to one

        Returns:
            Self after normalization

        """
        s = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i, :] /= s[i]
        return self

    def assign_col(self: Prob, i: int, col: numpy.ndarray):
        """
        Replace a column of self with data specified by the arguments

        Args:
            i: Column index
            col: Column value

        Returns:
            Self after assignment
        """
        self[:, i] = col
        return self

    def likelihoods(self: Prob, v: numpy.ndarray) -> numpy.ndarray:
        r"""Likelihoods for vector of data

        Args:
            v: A time series of integer observations

        Returns:
            2-d array of state likelihoods

        If self represents probability of observing integers given
        state, ie, self[s, y] = Probability(observation=y|state=s),
        then this function returns the likelihood for each state given
        the observation at a particular time.  Given T = len(v) and
        self.shape = (M,N), this returns L with L.shape = (T,M) and L[t,a] =
        Prob(v[t]|a) \forall t \in [0,T) and a in [0,M).

        """
        return self[:, v].T

    def cost(self: Prob, nu: numpy.ndarray, py: numpy.ndarray):
        """Efficient calculation of numpy.outer(nu, py)*self (where * is
        element-wise)

        Args:
            nu:  Cost of minimum cost path to each state
            py: Likelihood of each state given data y[t]
        Returns:
            Minimum costs for sequences ending in state pairs

        Used in Viterbi decoding with self[a,b] =
        Prob(s[t+1]=b|s[t]=a).  If nu[a] = minimum cost of s[t-1]=a
        given the data y_0^{t-1} and py[b] = Probability observation =
        y[t] given s[t]=b, then this method returns a 2-d array, C,
        with C[a,b] = cost of minimum cost path ending with s[t-1]=a,
        s[t]=b given observations y_0^t.

        """
        return (self.T * nu).T * py

    def step_forward(self: Prob, alpha: numpy.ndarray):
        """Replace values of argument a with matrix product a*self.

        Args:
            alpha (numpy.ndarray):  Alpha[t]

        Used in forward algorithm.  In the vector argument
        alpha[a]=Probability(s[t]=a|y_0^t).  The resulting value is a
        vector A with A[a] = Probability(s[t+1]=a|y_0^t).

        Not done inline because c version is better than inline

        """
        alpha[:] = numpy.dot(alpha, self)

    def step_back(self: Prob, b: numpy.ndarray):
        """Replace values of argument a with matrix product self*a

        Args:
            b: See b[t] in the book

        Used in backward algorithm.  The vector result is beta[t-1]
        which is sort of like the vector alpha in step_forward.  See
        Chapter 2 of the book for a precise explanation.  The
        argument, b, already includes the probability of the
        observation at time t.  The calculation here applies the
        conditional state probability matrix backwards.

        Not done inline because c version is better than inline

        """
        b[:] = numpy.dot(self, b)

    def values(self: Prob) -> Prob:
        """
        Produce values of self

        Returns:
            self

        This is a hack to free subclasses from the requirement of self
        being an nd_array

        """
        return self


# --------------------------------
# Local Variables:
# mode: python
# End:
