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
        mu: numpy.ndarray Means; a value for each state
        variance: numpy.ndarray Variances; a value for each state.
        rng: Generator with state

    The probability of observation y given state s is:
    Prob(y|s) = 1/\sqrt(2\pi var[s]) exp(-(y-mu[s])^2/(2*var[s])

    """
    _parameter_keys = "mu variance".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: Gauss,
        mu: numpy.ndarray,
        variance: numpy.ndarray,
        rng: numpy.random.Generator,
    ):
        assert len(variance) == len(mu)
        self.mu = mu
        self.variance = variance
        self.sigma = numpy.sqrt(variance)
        self._rng = rng
        self.dtype = [numpy.float64]
        self.norm = 1 / numpy.sqrt(2 * numpy.pi * variance)
        self.n_states = len(variance)

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
        """Calculate and return likelihoods of states for all observations.

        Returns:
            self.p_y with self.p_y[t,s] = Prob(y[t]|state[t]=s)

        """
        assert self._y.reshape((-1, 1)).shape == (self.n_times, 1)
        d = self.mu - self._y.reshape((-1, 1))
        self._likelihood = numpy.exp(-d * d / (2 * self.variance)) * self.norm
        return self._likelihood

    # ToDo: Why do I need Any to make mypy happy?
    def diffs_to_var(self: Gauss, diffs: numpy.ndarray,
                     wsum: typing.Any) -> typing.Any:
        """ Formula for reestimating variance.

        Args:
            diffs
            wsum

        Returns:
            New estimate of variance

        Not in-line for convenience of MAP subclass
        """
        return (diffs * diffs).sum(axis=0) / wsum

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
        diffs = (self.mu - y.reshape((-1, 1))) * numpy.sqrt(w)
        self.variance = self.diffs_to_var(diffs, wsum)
        self.sigma = numpy.sqrt(self.variance)
        self.norm = 1 / numpy.sqrt(2 * numpy.pi * self.variance)


class GaussMAP(Gauss):
    """For the variance use maximum a posteriori probability estimation
    instead of maximum likelihood.  The prior is inverse gamma

    """

    def __init__(self: GaussMAP,
                 mu: numpy.ndarray,
                 variance: numpy.ndarray,
                 rng: numpy.random.Generator,
                 alpha: float = 1,
                 beta: float = 1):
        super().__init__(mu, variance, rng)
        self.alpha = alpha
        self.beta = beta

    def log_prior(self):
        return_value = 0.0
        for s in range(self.n_states):
            return_value += -(self.alpha + 1) * numpy.log(
                self.variance[s]) - self.beta / self.variance[s]
        return return_value

    def diffs_to_var(self: GaussMAP, diffs: numpy.ndarray, wsum: float):
        numerator = 2 * self.beta + (diffs * diffs).sum(axis=0)
        denominator = 2 * self.alpha + 2 + wsum
        return numerator / denominator


class MultivariateGaussian(hmm.base.Observation_0):
    """Observation model for vector measurements.

    Args:
        mu[n_states, dimension]: Mean of distribution for each state
        sigma[n_states, dimension, dimension]: Covariance matrix for each state
        rng: Random number generator with state
        nu: Inverse Wishart parameter
        Psi: Inverse Wishart parameter
        small: Raise error if total likelihood of any observation is
            less than small

    Implements Maximum a posteriori probability estimation of the
    variance of each state.  Without data, sigma[s,i,i] =
    \frac{Psi}{nu + dim +1} where dim is the dimension of the
    observations

    """
    _parameter_keys = "mu sigma".split()

    def __init__(  # pylint: disable = super-init-not-called
        self: MultivariateGaussian,
        mu: numpy.ndarray,
        sigma: numpy.ndarray,
        rng: numpy.random.Generator,
        psi: float = .1,
        Psi: typing.Union[numpy.ndarray, None] = None,
        nu: float = 4,
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
        for s in range(self.n_states):
            self.inverse_sigma[s, :, :] = numpy.linalg.inv(self.sigma[s, :, :])
            determinant = numpy.linalg.det(sigma[s, :, :])
            self.norm[s] = 1 / numpy.sqrt(
                (2 * numpy.pi)**self.dimension * determinant)
        self._rng = rng
        if Psi is None:
            Psi = numpy.eye(self.dimension) * psi
        self.Psi = Psi
        self.nu = nu
        self.small = small

    def random_out(  # pylint: disable = arguments-differ
            self: MultivariateGaussian, s: int) -> numpy.ndarray:
        return self._rng.multivariate_normal(self.mu[s], self.sigma[s])

    def __str__(self: MultivariateGaussian) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % self.__class__
        for s in range(self.n_states):
            rv += 'For state %d:\n' % s
            rv += ' inverse_sigma = \n%s\n' % self.inverse_sigma[s]
            rv += ' mu = %s' % self.mu[s]
            rv += ' norm = %f\n' % self.norm[s]
        numpy.set_printoptions(precision=save)
        return rv

    def calculate(self: MultivariateGaussian) -> numpy.ndarray:
        """
        Calculate and return likelihoods.

        Returns:
            self.p_y[t,s] = Prob(y[t]|state[t]=s)

        Assumes self.observe has assigned a single numpy.ndarray to self._y
        """
        assert self._y.shape == (
            self.n_times,
            self.dimension), 'You must call observe before calling calculate.'
        for t in range(self.n_times):
            for s in range(self.n_states):
                d = (self._y[t] - self.mu[s])
                dQd = float(numpy.dot(d, numpy.dot(self.inverse_sigma[s], d)))
                if dQd > 300:  # Underflow
                    self._likelihood[t, s] = 0
                else:
                    self._likelihood[t, s] = self.norm[s] * numpy.exp(-dQd / 2)
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
        for s in range(self.n_states):
            rrsum = numpy.zeros((self.dimension, self.dimension))
            for t in range(self.n_times):
                r = y[t] - self.mu[s]
                rrsum += w[t, s] * numpy.outer(r, r)
            self.sigma[s, :, :] = (self.Psi + rrsum) / (wsum[s] + self.nu +
                                                        self.dimension + 1)
            det = numpy.linalg.det(self.sigma[s])
            assert (det > 0.0)
            self.inverse_sigma[s, :, :] = numpy.linalg.inv(self.sigma[s])
            self.norm[s] = 1.0 / (numpy.sqrt(
                (2 * numpy.pi)**self.dimension * det))

    def log_prior(self):
        return_value = 0.0
        for s in range(self.n_states):
            log_det = numpy.log(numpy.linalg.det(self.sigma[s]))
            trace = numpy.dot(self.Psi, self.inverse_sigma[s]).trace()
            return_value += -(self.nu + self.dimension +
                              1) * log_det / 2 - trace / 2
        return return_value


class AutoRegressive(GaussMAP):
    r"""Scalar autoregressive model with Gaussian residuals

    Args:
        ar_coefficients[n_states, ar_order]: Auto-regressive coefficients
        variance[n_states]: Residual variance for each state
        rng: Random number generator with state
        alpha: Part of prior for variance
        beta: Part of prior for variance
        small: Throw error if likelihood at any time is less than small

    Model: likelihood[t,s] = Normal(mu_{t,s}, var[s]) at _y[t]
           where mu_{t,s} = ar_coefficients[s] \cdot _y[t-n_ar:t] + offset[s]

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
            alpha: float = 4.0,
            beta: float = 16.0,
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
        for s in range(self.n_states):
            self.norm[s] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[s])
        self._rng = rng
        self.alpha = alpha
        self.beta = beta
        self.small = small

    def initialize_out(self: AutoRegressive):
        """ Prepare for calls to self.random_out
        """
        self.history = numpy.zeros(self.ar_order + 1)
        self.history[-1] = 1.0

    # ToDo: Better way to have different arguments for method of
    # subclass without mypy or pylint complaints
    def random_out(  # type: ignore
            # pylint: disable = arguments-differ
            self: AutoRegressive,
            s: int) -> numpy.ndarray:
        mu = numpy.dot(self.history, self.ar_coefficients_offset[s])
        rv = self._rng.normal(mu, self.variance[s])
        self.history[:self.ar_order - 1] = self.history[1:self.ar_order]
        self.history[self.ar_order - 1] = rv
        return rv

    def __str__(self: AutoRegressive) -> str:
        save = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=3)
        rv = 'Model %s instance\n' % type(self)
        for s in range(self.n_states):
            rv += 'For state %d:\n' % s
            rv += ' variance = \n%s\n' % self.variance[s]
            rv += ' ar_coefficients = %s' % self.ar_coefficients_offset[s, :-1]
            rv += ' offset = %s' % self.ar_coefficients_offset[s, -1]
            rv += ' norm = %f\n' % self.norm[s]
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
            likelihood where likelihood[t,s] = Prob(y[t]|state[t]=s)

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

    # mypy objects: "incompatible with supertype Integerobservation"
    def reestimate(  # type: ignore
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

        for s in range(self.n_states):
            w_y = w1[:, s] * self._y
            w_context = (w1[:, s] * self.context.T).T
            # pylint: disable = unused-variable
            fit, residuals, rank, singular_values = numpy.linalg.lstsq(
                w_context, w_y, rcond=None)
            assert rank == self.ar_order + 1
            self.ar_coefficients_offset[s, :] = fit
            delta = w_y - numpy.inner(w_context, fit)
            sum_squared_error = numpy.inner(delta, delta)
            numerator = 2 * self.beta + numpy.inner(delta, delta)
            denominator = 2 * self.alpha + 2 + wsum[s]
            self.variance[s] = numerator / denominator
            self.norm[s] = 1 / numpy.sqrt(2 * numpy.pi * self.variance[s])


# --------------------------------
# Local Variables:
# mode: python
# End:
