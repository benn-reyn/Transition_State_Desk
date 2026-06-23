import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class WeightedGARCH:
    """
    Regime-Specific GARCH(1,1) for a single asset.
    Fits independent GARCH parameters per regime using true unblended
    residuals, utilizing smoothed probabilities as likelihood weights.
    """
    def __init__(self):
        self.params = {}          # {0: (omega, alpha, beta), 1: (omega, alpha, beta)}
        self.h = {}               # {0: array(T,), 1: array(T,)} conditional variances

    def _garch_likelihood(self, params, resid, weights):
        """Weighted negative log‑likelihood for a single regime track."""
        omega, alpha, beta = params
        T = len(resid)

        # Strict stationarity constraint
        if alpha + beta >= 0.999:
            return 1e10

        # Initialize variance using the periods when this specific regime was active
        mask = weights > 0.05
        init_var = np.var(resid[mask]) if mask.sum() > 5 else np.var(resid)
        init_var = max(init_var, 1e-6)

        h = np.zeros(T)
        h[0] = init_var
        loglik = 0.0

        # Continuous recursion allows the GARCH engine to maintain historical memory
        for t in range(1, T):
            h[t] = omega + alpha * resid[t-1]**2 + beta * h[t-1]
            
            # Bound variance to avoid log(0) errors
            h[t] = max(h[t], 1e-8)
            
            # Apply weights strictly to the loss function evaluation
            if weights[t] > 1e-8:
                loglik += weights[t] * norm.logpdf(resid[t], scale=np.sqrt(h[t]))

        return -loglik

    def fit(self, resid_calm, resid_bear, smoothed_probs):
        """Fits both GARCH tracks simultaneously using dedicated residual profiles."""
        T = len(resid_calm)
        

        # Standard loose bounds for an asset spending most of its time active
        initial_guess_calm = [1e-5, 0.05, 0.90]
        bounds_calm = [(1e-10, None), (1e-10, 0.30), (1e-10, 0.98)]
        
        res_0 = minimize(
            self._garch_likelihood, initial_guess_calm, 
            args=(resid_calm, smoothed_probs[:, 0]), 
            bounds=bounds_calm, method='L-BFGS-B'
        )
        
        # Force alpha and beta away from zero so the optimizer cannot cheat 
        # with a flat constant line during data-sparse regimes.
        initial_guess_bear = [1e-4, 0.15, 0.75]
        bounds_bear = [(1e-10, None), (0.02, 0.45), (0.50, 0.95)]
        
        res_1 = minimize(
            self._garch_likelihood, initial_guess_bear, 
            args=(resid_bear, smoothed_probs[:, 1]), 
            bounds=bounds_bear, method='L-BFGS-B'
        )
        
        #optimal params
        self.params[0] = res_0.x
        self.params[1] = res_1.x
        
        #  historical conditional variance tracks
        for k, resid_track in enumerate([resid_calm, resid_bear]):
            omega, alpha, beta = self.params[k]
            mask = smoothed_probs[:, k] > 0.05
            init_var = np.var(resid_track[mask]) if mask.sum() > 5 else np.var(resid_track)
            
            h_track = np.zeros(T)
            h_track[0] = max(init_var, 1e-6)
            for t in range(1, T):
                h_track[t] = omega + alpha * resid_track[t-1]**2 + beta * h_track[t-1]
            self.h[k] = h_track

    def forecast(self, e_last, h_last, xi_now):
        """Blends next-day expectations using current filtered probabilities."""
        h_next = {}
        for k in [0, 1]:
            omega, alpha, beta = self.params[k]
            h_next[k] = omega + alpha * e_last**2 + beta * h_last[k]
        return xi_now[0] * h_next[0] + xi_now[1] * h_next[1]