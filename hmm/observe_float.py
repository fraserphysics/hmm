"""observe_float.py: Various observation models for floating point measurements.

Use these models with the Hidden Markov Models in base.py.

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
        mu: numpy.ndarray,
        var: numpy.ndarray,
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
    def random_out(  # type: ignore
            # pylint: disable = arguments-differ
            self: Gauss,
            s: int) -> float:
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
    def reestimate(  # type: ignore
            # pylint: disable = arguments-differ
            self: Gauss,
            w: numpy.ndarray):
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


# Todo: Test this class
class MultivariateGaussian(hmm.base.Observation_0):
    """Observation model for vector measurements.

    Args:
        mu[n_states, dimension]: Mean of distribution for each state
        sigma[n_states, dimension, dimension]: Covariance matrix for each state
        rng: Random number generator with state
        inverse_wishart_a: Part of prior for covariance matrix
        inverse_wishart_b: Part of prior for covariance matrix
        small: Raise error if total likelihood of any observation is
            less than small

    Without data, sigma[s,i,i] = b/a

    """
    _parameter_keys = "mu sigma".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: MultivariateGaussian,
        mu: numpy.ndarray,
        sigma: numpy.ndarray,
        rng: numpy.random.Generator,
        inverse_wishart_a=4,
        inverse_wishart_b=0.1,
        small=1.0e-100,
    ):
        # Check arguments
        self.n_states, self.dimension = mu.shape
        assert sigma.shape == (self.n_states, self.dimension, self.dimension)
        assert isinstance(rng, numpy.random.Generator)

        # Assign arguments to self
        self.mu = mu
        self.sigma = sigma
        self.inverse_sigma = numpy.empty(
            (self.n_states, self.dimension, self.dimension))
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.inverse_sigma[i, :, :] = numpy.linalg.inv(self.sigma[i, :, :])
            determinant = numpy.linalg.det(sigma[i, :, :])
            self.norm[i] = 1 / numpy.sqrt(
                (2 * numpy.pi)**self.dimension * determinant)
        self._rng = rng
        self.inverse_wishart_a = inverse_wishart_a
        self.inverse_wishart_b = inverse_wishart_b
        self.small = small

    def random_out(  # pylint: disable = arguments-differ
            self: MultivariateGaussian, s: int) -> numpy.ndarray:
        raise RuntimeError(
            'random_out not implemented for MultivariateGaussian')

    def __str__(self: MultivariateGaussian) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % self.__class__
        for i in range(self.n_states):
            rv += 'For state %d:\n' % i
            rv += ' inverse_sigma = \n%s\n' % self.inverse_sigma[i]
            rv += ' mu = %s' % self.mu[i]
            rv += ' norm = %f\n' % self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def calculate(self: MultivariateGaussian) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,i] = P(y(t)|s(t)=i)

        Assumes self.observe has assigned a single numpy.ndarray to self._y
        """
        assert self._y.shape == (
            self.n_times,
            self.dimension), 'You must call observe before calling calculate.'
        for t in range(self.n_times):
            for i in range(self.n_states):
                d = (self._y[t] - self.mu[i])
                dQd = float(numpy.dot(d, numpy.dot(self.inverse_sigma[i], d)))
                if dQd > 300:  # Underflow
                    self._likelihood[t, i] = 0
                else:
                    self._likelihood[t, i] = self.norm[i] * numpy.exp(-dQd / 2)
            if self._likelihood[t, :].sum() < self.small:
                raise ValueError("""Observation is not plausible from any state.
self.likelihood[{0},:]={1}""".format(t, self._likelihood[t, :]))
        return self._likelihood

    def reestimate(
        self: MultivariateGaussian,
        w: numpy.ndarray,
    ):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights; Prob(state[t]=s) given data and
                old model

        """
        y = self._y
        wsum = w.sum(axis=0)
        self.mu = (numpy.inner(y.T, w.T) / wsum).T
        # Inverse Wishart prior parameters.  Without data self.sigma = b/a
        for i in range(self.n_states):
            rrsum = numpy.zeros((self.dimension, self.dimension))
            for t in range(self.n_times):
                r = y[t] - self.mu[i]
                rrsum += w[t, i] * numpy.outer(r, r)
            self.sigma = (self.inverse_wishart_b * numpy.eye(self.dimension) +
                          rrsum) / (self.inverse_wishart_a + wsum[i])
            det = numpy.linalg.det(self.sigma)
            assert (det > 0.0)
            self.inverse_sigma[i, :, :] = numpy.linalg.inv(self.sigma)
            self.norm[i] = 1.0 / (numpy.sqrt(
                (2 * numpy.pi)**self.dimension * det))


# Todo: Test this class
class AutoRegressive(hmm.base.Observation_0):
    r"""Scalar autoregressive model with Gaussian residuals

    Args:
        ar_coefficients[n_states, ar_order]: Auto-regressive coefficients
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state
        inverse_wishart_a: Part of prior for variance
        inverse_wishart_b: Part of prior for variance
        small: Throw error if likelihood at any time is less than small

    Model: likelihood[t,i] = Normal(mu_{t,i}, var[i]) at _y[t]
           where mu_{t,i} = ar_coefficients[i] \cdot _y[t-n_ar:t] + offset[i]

           In method calculate, likelihoods are set to zero for t
           closer to starting segment boundaries than AR-order because
           not enough prior measurements exist.

    """
    _parameter_keys = "ar_coefficients offset variance".split()

    def __init__(  # pylint: disable = super-init-not-called
            self: AutoRegressive,
            ar_coefficients: numpy.ndarray,
            offset: numpy.ndarray,
            variance: numpy.ndarray,
            rng: numpy.random.Generator,
            inverse_wishart_a: float = 4.0,
            inverse_wishart_b: float = 16.0,
            small: float = 1.0e-100):
        assert len(variance.shape) == 1
        assert len(offset.shape) == 1
        assert len(ar_coefficients.shape) == 2

        self.n_states, self.ar_order = ar_coefficients.shape

        assert offset.shape[0] == self.n_states
        assert variance.shape[0] == self.n_states
        assert isinstance(rng, numpy.random.Generator)

        # Store offset in self.ar_coefficients_offset for convenience in
        # both calculating likelihoods and in re-estimation
        self.ar_coefficients_offset = numpy.empty(
            (self.n_states, self.ar_order + 1))
        self.ar_coefficients_offset[:, :self.ar_order] = ar_coefficients
        self.ar_coefficients_offset[:, self.ar_order] = offset

        self.variance = variance
        self.norm = numpy.empty(self.n_states)
        for i in range(self.n_states):
            self.norm[i] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[i])
        self._rng = rng
        self.inverse_wishart_a = inverse_wishart_a
        self.inverse_wishart_b = inverse_wishart_b
        self.small = small

    def random_out(  # pylint: disable = arguments-differ
            self: AutoRegressive, s: int) -> numpy.ndarray:
        raise RuntimeError('random_out not implemented for AutoRegressive')

    def __str__(self: AutoRegressive) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % type(self)
        for i in range(self.n_states):
            rv += 'For state %d:\n' % i
            rv += ' variance = \n%s\n' % self.variance[i]
            rv += ' ar_coefficients = %s' % self.ar_coefficients_offset[i, :-1]
            rv += ' offset = %s' % self.ar_coefficients_offset[i, -1]
            rv += ' norm = %f\n' % self.norm[i]
        numpy.set_printoptions(precision=save)
        return rv

    def _concatenate(self: AutoRegressive, y_segs: typing.Sequence) -> tuple:
        """Attach context to self and return the modified concatenated data
        and segment information.

        Args:
            y_segs: Independent measurement sequences.  Each sequence
            is a 1-d numpy array.

        Returns:
            (all_data, Segment boundaries)

        This method shortens each segment by ar_order.  That enables
        having a true context for each element of self._y

        self.context will be used in calculate and reestimate.
        After values get assigned, context[t, :-1] = previous
        observations, and context[t, -1] = 1.0

        """
        length = 0
        t_seg = [0]
        for seg in y_segs:
            length += len(seg) - self.ar_order
            t_seg.append(length)
        all_data = numpy.empty(length)
        self.context = numpy.ones((length, self.ar_order + 1))
        for i in range(len(t_seg) - 1):
            all_data[t_seg[i]:t_seg[i + 1]] = y_segs[i][self.ar_order:]
            for delta_t in range(0, t_seg[i + 1] - t_seg[i]):
                self.context[t_seg[i] +
                             delta_t, :-1] = y_segs[i][delta_t:delta_t +
                                                       self.ar_order]
                # The one in the last place of context gets multiplied
                # by the last elements of self.ar_coefficients_offset
                # in self.calculate().  It is an offset term.
        return all_data, t_seg

    def calculate(self: AutoRegressive) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            likelihood where likelihood[t,i] = Prob(y(t)|s(t)=i)

        """
        assert self._y.shape == (self.n_times,)

        for t in range(self.n_times):
            delta = self._y[t] - numpy.dot(self.ar_coefficients_offset,
                                           self.context[t])
            exponent = -delta * delta / (2 * self.variance)
            if exponent.min() < -300:  # Underflow
                for s in range(self.n_states):
                    if exponent[s] < -300:
                        self._likelihood[t, s] = 0.0
                    else:
                        self._likelihood[t, s] = self.norm[s] * numpy.exp(
                            -delta[s] * delta[s] / (2 * self.variance[s]))
            else:
                self._likelihood[t, :] = self.norm * numpy.exp(
                    -delta * delta / (2 * self.variance))

            if self._likelihood[t, :].sum() < self.small:
                raise ValueError("""Observation is not plausible from any state.
self.likelihood[{0},:]={1}""".format(t, self._likelihood[t, :]))
        return self._likelihood

    def reestimate(
        self: AutoRegressive,
        w: numpy.ndarray,
    ):
        """
        Estimate new model parameters.  self._y already assigned

        Args:
            w: Weights. w[t,s] = Prob(state[t]=s) given data and
                old model

        """
        mask = w >= self.small  # Small weights confuse the residual
        # calculation in least_squares()
        w2 = mask * w
        wsum = w2.sum(axis=0)
        w1 = numpy.sqrt(w2)  # n_times x n_states array of weights

        for i in range(self.n_states):
            w_y = w1[:, i] * self._y
            w_context = (w1[:, i] * self.context.T).T
            # pylint: disable = unused-variable
            fit, residuals, rank, singular_values = numpy.linalg.lstsq(
                w_context, w_y, rcond=None)
            assert rank == self.ar_order + 1
            self.ar_coefficients_offset[i, :] = fit
            delta = w_y - numpy.inner(w_context, fit)
            sum_squared_error = numpy.inner(delta, delta)
            self.variance[i] = (self.inverse_wishart_b + sum_squared_error) / (
                self.inverse_wishart_a + wsum[i])
            self.norm[i] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[i])


# --------------------------------
# Local Variables:
# mode: python
# End:
