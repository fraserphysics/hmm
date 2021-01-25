""" base.py: Implements basic HMM algorithms.

Classes:

    :py:class:`Observation`:
        Models of discrete observations

    :py:class:`HMM`:
        A Hidden Markov Model implementation
"""
from __future__ import annotations  # Enables, eg, (self: HMM,

import typing  # For type hints

import numpy
import numpy.random

COPYRIGHT = """
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
        y_mod : Probability model for observations
        rng : Generator with state

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

    def forward(self) -> float:  # HMM instance
        """Recursively calculate state probabilities.

        Returns:
            Average log (base e) likelihood per point of entire observation sequence

        Requires that observation probabilities have already been calculated

        On entry:

        - self                is an HMM

        - self.p_y_by_state   has been calculated

        - self.alpha          has been allocated

        - self.gamma_inv      has been allocated

        On return:

        - 1/self.gamma_inv[t] = Pr{y(t)=y(t)|y_0^{t-1}}
        - self.alpha[t, i] = Pr{s(t)=i|y_0^t}

        """

        last = numpy.copy(self.p_state_initial.reshape(-1))  # Copy
        for t in range(len(self.p_y_by_state)):
            last *= self.p_y_by_state[t]  # Element-wise multiply
            self.gamma_inv[t] = 1 / last.sum()
            last *= self.gamma_inv[t]
            self.alpha[t, :] = last
            self.p_state2state.step_forward(last)
        return -(numpy.log(self.gamma_inv)).sum()  # End of forward()

    def backward(self):  # HMM instance
        """
        Baum Welch backwards pass through state conditional likelihoods.


        Calculates values of self.beta which "reestimate()" needs.

        On entry :

        - self               is an HMM

        - self.p_y_by_state  has been calculated

        - self.gamma_inv     has been calculated by forward

        - self.beta          has been allocated

        On return:

        - For each state i, beta[t, i] = Pr{y_{t+1}^T|s(t)=i}/Pr{y_{t+1}^T}

        """
        last = numpy.ones(self.n_states)
        # iterate backwards through y
        for t in range(len(self.p_y_by_state) - 1, -1, -1):
            self.beta[t, :] = last
            last *= self.p_y_by_state[t]
            last *= self.gamma_inv[t]
            self.p_state2state.step_back(last)
        # End of backward()

    def train(
            self: HMM,
            y,  # Type matches self.y_mod
            n_iter: int = 1,
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

        log_like_list = []
        self.n_y = self.y_mod.observe(y)
        assert self.n_y > 1
        # Ensure allocation and size of alpha and gamma_inv
        self.alpha = numpy.empty((self.n_y, self.n_states))
        self.beta = numpy.empty((self.n_y, self.n_states))
        self.gamma_inv = numpy.empty((self.n_y,))
        for it in range(n_iter):
            self.p_y_by_state = self.y_mod.calculate()
            log_likelihood_per_step = self.forward() / len(self.p_y_by_state)
            if display:
                print("it= %d LLps= %7.3f" % (it, log_likelihood_per_step))
            log_like_list.append(log_likelihood_per_step)
            self.backward()
            self.reestimate()
        return log_like_list  # End of train()

    def reestimate(self: HMM):
        """Phase of Baum Welch training that reestimates model parameters

        Based on previously calculated values in self.alpha and
        self.beta, the code here updates state transition
        probabilities and initial state probabilities.  Contains a
        call to the y_mod.reestimate method that updates observation
        model parameters.

        """

        u_sum = numpy.zeros((self.n_states, self.n_states), numpy.float64)
        # u_sum[i,j] = \sum_{t:gamma[t]>0} alpha[t,i] * beta[t+1,j] *
        # p_y_by_state[t+1]/gamma[t+1]
        u_sum = numpy.einsum(
            'ti,tj,tj,t->ij',  # Specifies the i,j indices and sum over t
            self.alpha[:-1],  # indices t,i
            self.beta[1:],  # indices t,j
            self.p_y_by_state[1:],  # indices t,j
            self.gamma_inv[1:]  # index t
        )
        self.alpha *= self.beta  # Saves allocating new array for result
        alpha_beta = self.alpha
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
                self.p_y_by_state.shape was assigned externally.

        Returns:
            Maximum likelihood state sequence

        This implements the Viterbi algorithm.
        """

        if not (y is None):
            # Calculate likelihood of data given state
            self.n_y = self.y_mod.observe(y)
            self.p_y_by_state = self.y_mod.calculate()
        n_y, n_states = self.p_y_by_state.shape
        assert self.n_states == n_states
        assert n_y > 1

        # Allocate working memory
        best_predecessors = numpy.empty((self.n_y, self.n_states),
                                        numpy.int32)  # Best predecessors
        ss = numpy.ones((self.n_y, 1), numpy.int32)  # State sequence

        # Use initial state distribution for first best_path_cost
        best_path_cost = self.p_y_by_state[0] * self.p_state_initial

        for t in range(1, self.n_y):
            # p_state2state*outer(nu, p_y_by_state[t])
            cost = self.p_state2state.cost(best_path_cost, self.p_y_by_state[t])
            best_predecessors[t] = cost.argmax(axis=0)
            best_path_cost = numpy.choose(best_predecessors[t], cost)
            assert best_path_cost.max() > 0
            best_path_cost /= best_path_cost.max()  # Prevent underflow

        # Backtrack from best end state
        last_s = numpy.argmax(best_path_cost)
        for t in range(self.n_y - 1, -1, -1):
            ss[t] = last_s
            last_s = best_predecessors[t, last_s]
        return ss.flat  # End of decode()

    def initialize_y_model(
            self: HMM,
            y,  #  Match self.y_mod
            s_t: typing.Optional[numpy.ndarray] = None):
        """ Given data, make plausible y_model.

        Args:
            y: Observation sequence
            s_t: State sequence

        """
        n_y = self.y_mod.observe(y)
        if s_t is None:
            s_t = numpy.array(self.state_simulate(n_y), numpy.int32)
        alpha = numpy.zeros((n_y, self.n_states))
        t = numpy.arange(n_y)
        alpha[t, s_t] = 1
        self.alpha = alpha
        self.beta = alpha.copy()
        self.gamma_inv = numpy.ones(n_y)
        self.p_y_by_state = numpy.ones((n_y, self.n_states))
        self.reestimate()

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

        """

        self.p_y_by_state = self.rng.random((length, self.n_states))
        self.n_y = length
        if mask is not None:
            self.p_y_by_state *= mask

        return self.decode(None)  # End of state_simulate()

    def simulate(
        self: HMM,
        length: int,
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Generate a random sequence of observations of a given length.

        Args:
            length: Number of time steps to simulate

        Returns:
            states, where states[t] is the state at time t, and
                outs

        """

        # Initialize lists
        outs = []
        states = []
        # Set up cumulative distributions
        cum_init = numpy.cumsum(self.p_state_time_average[0])
        cum_tran = numpy.cumsum(self.p_state2state.values(), axis=1)

        # cum_rand generates random integers from a cumulative distribution
        def cum_rand(cum):
            return numpy.searchsorted(cum, self.rng.random())

        # Select initial state
        i = cum_rand(cum_init)
        # Select subsequent states and call model to generate observations
        for t in range(length):
            states.append(i)
            outs.append(self.y_mod.random_out(i))
            i = cum_rand(cum_tran[i])
        return (states, outs)

    def link(self: HMM, here: int, there: int, p: float):
        """Create (or remove) a link between state "here" and state "there".

        Args:
            here: One index of element to modify
            there: Other index of element to modify
            p: Weight or probability of linke

        The strength of the link is a function of both the argument
        "p" and the existing conditional probabilities of state
        transitions, self.p_state2state, in which
        self.p_state2state[here, there] is the probability of going to
        state there given that the system is in state here.  The code
        sets p_state2state[here, there] to p and then re-normalizes.
        Set self.p_state2state itself if you need to set exact values.
        You can use this method to modify topology before training.

        """
        self.p_state2state[here, there] = p
        self.p_state2state[here, :] /= self.p_state2state[here, :].sum()

    def __str__(self) -> str:  # HMM instance
        #save = numpy.get_printoptions
        # numpy.set_printoptions(precision=3)
        rv = """
%s with %d states
p_state_initial         = %s
p_state_time_average = %s
p_state2state =
%s
%s""" % (
            self.__class__,
            self.n_states,
            self.p_state_initial,
            self.p_state_time_average,
            self.p_state2state.values(),
            self.y_mod,
        )
        # numpy.set_printoptions(save)
        return rv[1:-1]


class Observation:
    """ Probability models for discrete observations from a finite set.

    Args:
        parameters: A dict {'p_ys': array}. Use dict for flexible sub-classes
        rng: A numpy.random.Generator for simulation

    Public methods and attributes:

    __init__

    observe

    random_out

    calculate

    reestimate

    model_py_state

    """

    def __init__(self: Observation, model_py_state: numpy.ndarray,
                 rng: numpy.random.Generator):
        self.model_py_state = model_py_state
        self._rng = rng
        self.n_states = self._normalize()
        self._observed_py_state = None

    def _normalize(self: Observation):
        """ Separate from __init__ to make subclass easy
        """
        self.model_py_state = Prob(self.model_py_state)
        self._cummulative_y = numpy.cumsum(self.model_py_state, axis=1)
        return len(self.model_py_state)

    def observe(self: Observation, ys) -> int:
        """ Attach measurement sequence[s] to self.

        Args:
            ys: A sequence of integer observations

        Returns:
            Length of observation sequence
        """
        self._y = ys
        self.n_y = len(self._y)
        self._observed_py_state = numpy.empty((self.n_y, self.n_states),
                                              dtype=numpy.float64)
        return self.n_y

    def calculate(self: Observation) -> numpy.ndarray:
        r"""
        Calculate likelihoods: self._observed_py_state[t,i] = P(y(t)|s(t)=i)

        Returns:
            p_y_by_state[t,i] \forall t \in [0,len(y)) and i \in [0,n_states)

        Assumes a previous call to measure has assigned self._y and allocated
            self._observed_py_state

        """

        # mypy: Unsupported target for indexed assignment ("None")
        self._observed_py_state[:, :] = self.model_py_state.likelihoods(  #  type: ignore
            self._y)
        return self._observed_py_state

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
        Estimate new model_py_state

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

        assert self._y.dtype == numpy.int32 and self._y.shape == (self.n_y,)

        # Loop over range of allowed values of y
        for yi in range(self.model_py_state.shape[1]):
            # yi was observed at times: numpy.where(self._y == yi)[0]
            # w.take(...): the conditional state probabilities at those times
            self.model_py_state.assign_col(
                yi,
                w.take(numpy.where(self._y == yi)[0], axis=0).sum(axis=0))
        self.model_py_state.normalize()
        self._cummulative_y = numpy.cumsum(self.model_py_state, axis=1)


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
        Make each row a probability that sums to one

        Returns:
            Self after normalization

        """
        s = self.sum(axis=1)
        for i in range(self.shape[0]):
            self[i, :] /= s[i]
        return self

    def assign_col(self: Prob, i: int, col: numpy.ndarray):
        """
        Replace column of self with data specified by the arguments

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

    # Todo: delete and do inline
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
