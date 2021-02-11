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


class Gauss(hmm.base.IntegerObservation):
    r"""Scalar Gaussian observation model

    Args:
        pars: (mu, var) (mu: numpy.ndarray) Means; a value for
            each state, (var: (numpy.ndarray)) Variances; a value for
            each state.
        rng: Generator with state

    The probability of obsrevation y given state s is:
    P(y|s) = 1/\sqrt(2\pi var[s]) exp(-(y-mu[s])^2/(2var[s])

    """
    _parameter_keys = "mu var".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: Gauss,
        mu: float,
        var: float,
        rng: numpy.random.Generator,
    ):
        assert len(var) == len(mu)
        self.mu = mu
        self.var = var
        self.sigma = numpy.sqrt(var)
        self._rng = rng
        self.dtype = [numpy.float64]
        self.norm = 1 / numpy.sqrt(2 * numpy.pi * var)
        self.n_states = len(var)

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
        self._likelihood = numpy.exp(-d * d / (2 * self.var)) * self.norm
        return self._likelihood

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
