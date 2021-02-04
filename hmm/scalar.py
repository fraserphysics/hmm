""" scalar.py: Scalar observation models for the Hidden Markov Models in base.py.
"""
# pylint: disable = attribute-defined-outside-init

from __future__ import annotations  # Enables, eg, (self: HMM,

import typing

import numpy

import hmm.base

COPYRIGHT = """Copyright 2021 Andrew M. Fraser

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


def make_random(shape: tuple, rng: numpy.random.Generator) -> hmm.base.Prob:
    """
    Make a random conditional distribution of given shape

    Args:
        shape:  Shape of resulting Prob instance
        rng: Holds state of pseudo-random
            number generator

    """
    return hmm.base.Prob(rng.uniform(0, 1, shape)).normalize()


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

    model_py_state because it's name is in an argument that's a dict

    """
    _parameter_keys = set(('model_py_state',))

    def __init__(  # pylint: disable = super-init-not-called
            self: Observation, parameters: dict, rng: numpy.random.Generator):
        assert set(parameters.keys()) == self._parameter_keys
        for key, value in parameters.items():
            setattr(self, key, value)
        self._rng = rng
        self.n_states = self._normalize()
        self._observed_py_state = None
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
        self._observed_py_state = numpy.empty((len(self._y), self.n_states))
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
        for yi in range(self.model_py_state.shape[1]):
            self.model_py_state.assign_col(
                yi,
                w.take(numpy.where(self._y == yi)[0], axis=0).sum(axis=0))
        self.model_py_state.normalize()
        self._cummulative_y = numpy.cumsum(self.model_py_state, axis=1)


class Gauss(Observation):
    r"""Scalar Gaussian observation model

    Args:
        pars: (mu, var) (mu: numpy.ndarray) Means; a value for
            each state, (var: (numpy.ndarray)) Variances; a value for
            each state.
        rng: Generator with state

    The probability of obsrevation y given state s is:
    P(y|s) = 1/\sqrt(2\pi var[s]) exp(-(y-mu[s])^2/(2var[s])

    """
    _parameter_keys = set("mu var".split())

    def __init__(
        self: Gauss,
        pars: dict,
        rng: numpy.random.Generator,
    ):
        super().__init__(pars, rng)
        self.sigma = numpy.sqrt(self.var)
        self.dtype = [numpy.float64]

    def _normalize(self: Gauss):  # Must override super._normalize
        self.norm = 1 / numpy.sqrt(2 * numpy.pi * self.var)
        return len(self.var)

    def __str__(self: Gauss) -> str:
        return "    mu=%s\nvar=%s " % (self.mu, self.var)

    # Ignore: Super returned int
    def random_out(  # pylint: disable = arguments-differ
            self: Gauss, s: int) -> float:
        """ For simulation, draw a random observation given state s

        Args:
            s: Index of state

        Returns:
            Random observation drawn from distribution conditioned on state s

        """
        return (self._rng.normal(self.mu[s], self.sigma[s]))

    def calculate(self: Gauss,) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        """
        assert self._y.reshape((-1, 1)).shape == (self.n_times, 1)
        d = self.mu - self._y.reshape((-1, 1))
        self._observed_py_state = numpy.exp(-d * d / (2 * self.var)) * self.norm
        return self._observed_py_state

    # Ignore: Super has optional argument warn
    def reestimate(  # pylint: disable = arguments-differ
            self: Gauss, w: numpy.ndarray):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights; Prob(state[t]=s) given data and
                old model

        """
        assert len(self._y) > 0
        y = self._y
        wsum = w.sum(axis=0)
        self.mu = (w.T * y).sum(axis=1) / wsum
        d = (self.mu - y.reshape((-1, 1))) * numpy.sqrt(w)
        self.var = (d * d).sum(axis=0) / wsum
        self.sigma = numpy.sqrt(self.var)
        self.norm = 1 / numpy.sqrt(2 * numpy.pi * self.var)


# --------------------------------
# Local Variables:
# mode: python
# End:
