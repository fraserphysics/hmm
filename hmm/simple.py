""" simple.py: Implements basic HMM algorithms.

Classes:

    :py:class:`HMM`:
        A Hidden Markov Model implementation

    :py:class:`Observation`:
        Models of discrete observations

    :py:class:`Prob`:
        A subclass of ndarray for conditional probability matrices

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
        p_state_time_average : Time average distribution of states
        p_state2state : Probability of state given state:
            p_state2state[a, b] = Prob(s[1]=b|s[0]=a)
        y_mod : Instance of class for probabilities of observations
        rng : Numpy generator with state

    p_state_time_average is averaged over training data.  It is not
    the stationary distribution of p_state2state.

    Arguments are passed by reference and they are modified by some of
    the methods of HMM.

    By initializing with rng created by the caller with, eg
    numpy.random.default_rng(seed), one can ensure reproducible
    pseudo-random sequences and avoid using the same pseudo-random
    sequences in different parts of the code.

    """

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
        self.gamma_inv: numpy.ndarray = None
        self.alpha: numpy.ndarray = None
        self.beta: numpy.ndarray = None

    def forward(self: HMM) -> float:
        """Recursively calculate state probabilities.

        Returns:
            Log (base e) of likelihood of HMM given entire observation sequence

        Requires that observation probabilities have already been calculated

        On entry:

        - self                    is an HMM

        - self.state_likelihood   has been calculated

        - self.alpha              has been allocated

        - self.gamma_inv          has been allocated

        On return:

        - 1/self.gamma_inv[t] = Prob{y[t]=y[t]|y[0:t]}
        - self.alpha[t, i] = Prob{s[t]=i|y[0:t+1]}

        """

        # last is a conditional distribution of state probabilities.
        # What it is conditioned on changes as the calculations
        # progress.
        last = numpy.copy(self.p_state_initial.reshape(-1))  # Copy
        for t in range(len(self.state_likelihood)):
            last *= self.state_likelihood[t]  # Element-wise multiply
            assert last.sum() > 0  # Fails if data is impossible
            self.gamma_inv[t] = 1 / last.sum()
            last *= self.gamma_inv[t]
            self.alpha[t, :] = last
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

        - For each state i, beta[t, i] = Prob(y[t+1:T]|s[t]=i)/Prob(y[t+1:T])

        """
        # last and beta are analogous to last and alpha in forward(),
        # but the precise interpretations are more complicated.
        last = numpy.ones(self.n_states)
        for t in range(len(self.state_likelihood) - 1, -1, -1):
            self.beta[t, :] = last
            last *= self.state_likelihood[t] * self.gamma_inv[t]
            last[:] = numpy.dot(self.p_state2state, last)

    def calculate(self: HMM):
        """Prepares for calling forward if it is not called from train.

        """
        self.state_likelihood = self.y_mod.calculate()
        self.n_times = self.y_mod.n_times
        self.alpha = numpy.empty((self.n_times, self.n_states))
        self.gamma_inv = numpy.empty((self.n_times,))

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
        if delta / abs(ll) < -1.0e-15:  # Todo: Why not zero?
            iteration = len(log_likelihood_list)
            raise ValueError("""
Training is not monotonic: LLps[{0}]={1} and LLps[{2}]={3} difference={4}
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

        self.p_state_time_average = alpha_beta.sum(axis=0)  # type: ignore
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

        if y is not None:  # Calculate likelihood of data given state
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

        # Use initial state distribution for first best_path_utility
        best_path_utility = self.state_likelihood[0] * self.p_state_initial

        for t in range(1, self.n_times):
            # utility = p_state2state*outer(best_path_utility, state_likelihood[t])
            utility = (self.p_state2state.T *
                       best_path_utility).T * self.state_likelihood[t]
            best_predecessors[t] = utility.argmax(axis=0)
            best_path_utility = numpy.choose(best_predecessors[t], utility)
            if best_path_utility.max() == 0:
                raise ValueError(
                    "Attempt to decode impossible observation sequence")
            best_path_utility /= best_path_utility.max()  # Prevent underflow

        # Find the best end state
        previous_best_state = numpy.argmax(best_path_utility)

        # Backtrack through best_predecessors to find the best
        # sequence.
        for t in range(self.n_times - 1, -1, -1):
            best_state_sequence[t] = previous_best_state
            previous_best_state = best_predecessors[t, previous_best_state]
        return best_state_sequence.reshape(-1)

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
        0.  The function does not use an observation function and does
        not simulate observations.  The simulate method provides draws
        of states and observations from a fully specified model.

        """

        self.state_likelihood = self.rng.random((length, self.n_states))
        self.n_times = length
        if mask is not None:
            self.state_likelihood *= mask

        try:
            state_sequence = self.decode(None)
        except ValueError as e:
            raise Exception(
                "State_simulate given an impossible mask constraint") from e

        return state_sequence

    def simulate(
        self: HMM,
        length: int,
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Generate a random draw from the model of sequences of states and
        observations of a given length.

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
        return numpy.array(states), numpy.array(outs)

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
        sets p_state2state[here, there] to p and then re-normalizes
        self.p_state2state.  Set self.p_state2state itself if you need
        to set exact values.  You can use this method to modify the
        state topology by removing links before training.

        """
        self.p_state2state[here, there] = p
        self.p_state2state[here, :] /= self.p_state2state[here, :].sum()

    def __str__(self: HMM) -> str:  # HMM instance
        return """{0} with {1:d} states
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

    def deallocate(self: HMM) -> HMM:
        """ Remove arrays assigned by train.

        To be called before writing a model to disk
        """
        del (self.alpha)
        del (self.beta)
        del (self.gamma_inv)
        return self


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

    n_states

    """

    def __init__(self: Observation,
                 py_state: numpy.ndarray,
                 rng: numpy.random.Generator = None):
        self._py_state = Prob(py_state)
        if rng is None:
            self._rng = numpy.random.default_rng()
        else:
            self._rng = rng
        self._cummulative_y = numpy.cumsum(self._py_state, axis=1)
        self.n_states = len(self._py_state)

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
        Calculate likelihoods: self._likelihood[t,s] = P(y[t]|state[t]=s)

        Returns:
            state_likelihood[t,s] \forall t \in [0:n_times] and s \in [0:n_states]

        Assumes a previous call to observe has assigned self._y and allocated
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

    def reestimate(self: Observation, w: numpy.ndarray):
        """
        Estimate new _py_state

        Args:
            w: w[t,s] = Prob(state[t]=s) given data and old model
        """

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
    """Subclass of ndarray for conditional probability matrices.  P[a,b]
    is the probability of b given a.  The class has additional methods
    and is designed to enable alternative implementations that run
    faster or in less memory but may be implemented by uglier code.

    """

    def __new__(cls, x: numpy.ndarray):
        """ Return a Prob instance of the argument.

        Args:
            x: An array of conditional probabilities

        """
        assert len(x.shape) == 2
        # cls is Prob.  This calls __new__ of numpy.ndarray and makes
        # the return value a Prob instance.
        return super().__new__(cls, x.shape, buffer=x.data)

    # See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def normalize(self: Prob) -> Prob:  # Prob instance
        """
        Modify self to make each row sum to one

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
        Prob(v[t]|a) \forall t \in [0:T] and a in [0:M].

        """
        return self[:, v].T

    def utility(self: Prob, nu: numpy.ndarray, py: numpy.ndarray):
        """Efficient calculation of numpy.outer(nu, py)*self (where * is
        element-wise)

        Args:
            nu:  Utility of maximum utility path to each state
            py: Likelihood of each state given data y[t]
        Returns:
            Maximum utilities for sequences ending in state pairs

        Used in Viterbi decoding with self[a,b] =
        Prob(s[t+1]=b|s[t]=a).  If nu[a] = maximum utility of s[t-1]=a
        given the data y[0:t] and py[b] = Probability observation =
        y[t] given s[t]=b, then this method returns a 2-d array, C,
        with C[a,b] = utility of maximum utility path ending with s[t-1]=a,
        s[t]=b given observations y[0:t+1].

        """
        return (self.T * nu).T * py

    def step_forward(self: Prob, alpha: numpy.ndarray):
        """Replace values of argument a with matrix product a*self.

        Args:
            alpha (numpy.ndarray):  Alpha[t]

        Used in forward algorithm.  In the vector argument
        alpha[a]=Probability(s[t]=a|y[0:t+1]).  The resulting value is a
        vector A with A[a] = Probability(s[t+1]=a|y[0:t+1]).

        Not done inline because function written in c is better

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

        Not done inline because function written in c is better

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
