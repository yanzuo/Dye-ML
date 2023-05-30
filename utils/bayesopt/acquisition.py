import numpy as np
from scipy.stats import norm
from utils.ml.models import GP


class AcquisitionFunction(object):
    """
    Base class for acquisition functions. Used to define the interface
    """

    def __init__(self, surrogate=None, verbose=False):
        self.surrogate = surrogate
        self.verbose = verbose

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class AcquisitionOnSubspace():
    def __init__(self, acq, free_idx, fixed_vals):
        self.acq = acq
        self.free_idx = free_idx
        self.fixed_vals = fixed_vals

    def evaluate(self, x: np.ndarray):
        x_fixed = [self.fixed_vals] * len(x)
        x_complete = np.hstack((np.vstack(x_fixed), x))
        return self.acq.evaluate(x_complete)


class EI(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, verbose=False):
        self.best = best
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "EI"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the EI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating EI at", x)
        mu, var = self.surrogate.predict(np.atleast_2d(x))
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu) / s

        return (s * gamma * norm.cdf(gamma) + s * norm.pdf(gamma)).flatten()


class PI(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function for a Gaussian model
    """

    def __init__(self, surrogate: GP, best: np.ndarray, tradeoff: float,
                 verbose=False):
        self.best = best
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"PI-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the PI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating PI at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu - self.tradeoff) / s
        return norm.cdf(gamma).flatten()


class UCB(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function for a Gaussian model
    """

    def __init__(self, surrogate: GP, tradeoff: float, verbose=False):
        self.tradeoff = tradeoff

        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"UCB-{self.tradeoff}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the UCB acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at
        """
        if self.verbose:
            print("Evaluating UCB at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        return -(mu - self.tradeoff * s).flatten()


class TS(AcquisitionFunction):
    """
    Thompson Sampling acquisition function for a Gaussian model
    """

    def __init__(self, surrogate: GP, verbose=False):
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "TS"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at
        """
        if self.verbose:
            print("Evaluating TS at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        return -np.random.normal(mu, s).flatten()

class MES(AcquisitionFunction):
    """
    Max-Value Entropy Search (MES) acquisition function for a Gaussian model:
    (https://arxiv.org/pdf/1703.01968.pdf)
    """

    def __init__(self, surrogate: GP, best: np.ndarray, verbose=False):
        self.best = best
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "MES"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the MES acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating MES at", x)
        mu, var = self.surrogate.predict(np.atleast_2d(x))
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu) / s

        pdfgamma = norm.pdf(gamma)
        cdfgamma = norm.cdf(gamma)

        return -(gamma * pdfgamma / (2 * cdfgamma + 1e-8) - np.log(cdfgamma + 1e-8)).flatten()


class MI(AcquisitionFunction):
    """
    Maximise Information (MI) acquisition function for a Gaussian model
    """

    def __init__(self, surrogate: GP,  verbose=False):
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return f"MI"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """
        Evaluates the MI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at
        """
        if self.verbose:
            print("Evaluating MI at", x)
        mu, var = self.surrogate.predict(x)
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        return s.flatten()