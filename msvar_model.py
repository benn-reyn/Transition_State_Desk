import numpy as np
from scipy.stats import multivariate_normal

class MSIAHVAR: 
    """
    Markov Switching Intercept Autoregressive Heteroskedastistic VAR(1)

    Params:
    - P: (K,K) transition matrix
    - intercepts: list of (N,) arrays
    - coefs : list of (N,N) autoregressive matrices
    - covs: list of (N,N) covariance matrices
    - filtered_probs: (T,K) real time regime probs
    - smoothed_probs: (T, K) full-sample regime probs
    
    Methods:
    - fit(data, max_iter, tol, shrinkage) : EM estimation
    - impulse_response(shock, horizon, xi0) : KvD closed‑form GIRF
    - unconditional_mean() : KvD unconditional mean
    - _hamilton_filter(Y, X) : forward recursion
    - _kim_smoother(filtered, predictive) : backward recursion
    - _ergodic_probs(P) : stationary distribution of Markov chain
    """
    def __init__(self, k_regimes=2):
        self.K = k_regimes
        self.N = None
        self.P = None
        self.intercepts = []
        self.coefs = []
        self.covs = []
        self.filtered_probs = None
        self.smoothed_probs = None

    def _ergodic_probs(self, P):
        """
        Stationary distribution of the Markov chain
        Used to initialize recursion with ergodic probs in hamilton filter and KvD mean
        Solves ξ* P = ξ*, ∑ ξ_i = 1
        """
        #K is 2 which is shape of 2x2 transition matrix
        K = P.shape[0]
        #takes P^T - I = 0 and removes redundant rows for AR coefs
        A = np.vstack(((P.T - np.eye(K))[:-1], np.ones(K)))
        #target vector for linear solver forced to sum to 1
        b = np.zeros(K); b[-1] = 1.0
        try:
            return np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            return np.ones(K) / K
    
    def _hamilton_filter(self, Y, X):
        """
        Overview: Hamilton filter recursively updates the highest
        probability for each step, will later require smoothing for efficiency
        compute filtered probs ξ_{t|t} and log-likelihood.
        Y: (T,N) observations at time t (log-returns)
        X: (T, N) lagged observations (t-1)
        returns: log-likelihood, filtered, predictive, densities
        """
        T, N = Y.shape
        K = self.K
        filtered = np.zeros((T, K)) #ξ_{t|t}
        predictive = np.zeros((T, K)) #ξ_{t|t-1}
        densities = np.zeros((T,K)) #η_{k,t}
        xi = self._ergodic_probs(self.P) #initial and ergodic state probs
        log_lik = 0.0

        #predict ξ_{t|t-1} = P' ξ_{t-1|t-1} 
        #for prediction of tomorrows regime based on returns characteristics of yesterday
        for t in range(T):
            xi_pred = self.P.T @ xi
            predictive[t] = xi_pred

            #compute regime conditional densities
            for k in range(K):
                # μ_{k,t} = c_k + A_k y_{t-1}
                mu = self.intercepts[k] + X[t] @self.coefs[k].T
                try:
                    # η_{k,t} = N(y_t | μ_{k,t}, Σ_k)
                    dens = multivariate_normal.pdf(
                        Y[t], mean=mu, cov=self.covs[k],
                        allow_singular=True
                    )
                except Exception:
                    dens = 1e-15
                densities[t,k] = max(dens, 1e-15) #don't want zero

            #update with Bayes rule (moved outside the k loop)
            # joint = ξ_{t|t-1} * η_{k,t}
            joint = xi_pred * densities[t]
            # marginal = ∑_j ξ_{t|t-1}(j) η_{j,t}
            marginal = joint.sum()
            if marginal <= 0 or np.isnan(marginal):
                marginal = 1e-15
                joint = np.ones(K) / K * marginal #ensure xi sums to 1
            # ξ_{t|t} = joint / marginal
            xi = joint / marginal
            filtered[t] = xi
            #add log-likelihood contribution
            log_lik += np.log(marginal)
            
        return log_lik, filtered, predictive, densities
        
    def _kim_smoother(self, filtered, predictive):
        """
        Overview: Given results from Hamilton, Kim Smoothing
        works backward from t to 1, determining the probabiility
        that where I ended up what is the most likely path that got
        me here? Kind of like weaponizing hindsight bias.
        Compute smoothed probs ξ_{t|T} using Kim (1994)
        filtered : (T,K), ξ_{t|t}
        predictive : (T,K), ξ_{t|t-1}
        Returns smoothed (T,K) and joint transitions (T,K,K).
        """
        T = len(filtered)
        K = self.K
        smoothed = np.zeros((T, K))
        smoothed[-1] = filtered[-1] #lookback from end
        joint_trans = np.zeros((T, K, K)) #joint prob i at t, j at t+1

        for t in range(T-2, -1, -1):
            for i in range(K):
                for j in range(K):
                    denom = predictive[t+1, j]
                    if denom == 0:
                        denom = 1e-15
                    #ξ_{t|T}(i) contribution from transition i to j
                    joint_trans[t+1, i, j] = (
                        filtered[t,i] * self.P[i,j] * smoothed[t+1, j] / denom
                    )
            #sum over j to get smooth prob at t
            smoothed[t] = joint_trans[t+1].sum(axis=1)
            #normalize
            smoothed[t] /= (smoothed[t].sum() + 1e-15)
    
        return smoothed, joint_trans
    
    def fit(self, data, max_iter=50, tol=1e-4, shrinkage=0.05):
        """
        Overview: In the MSIAH-VAR we've got two unknowns that are
        interdependent. The parameters (intercepts, betas ,volatility)
        and the states. We can't find one without the other. So we use
        the Expectation Maximization (EM) algorithm to alternate between them
        until we seen convergence. For instance, if we find our parameters are 
        correctish, what are the odds today was a Bull market? If we say 70%, 
        we then log that as a vote for 70% for todays regression (assuming we run OLS on observation
        probabilities found in the first step). This process iteratively provides stronger estimates.
        Estimate model parameters via Expectation Maximisation.
        data : array or DataFrame of log-returns (T rows, N columns)
        """
        if isinstance(data, np.ndarray):
            arr = data
        else:
            arr = data.values

        Y = arr[1:] #returns at time t
        X = arr[:-1] #lagged returns
        T, N = Y.shape
        self.N = N #number of variables
        K = self.K

        #init params, we assume sticky regimes as a basis
        self.P = np.full((K, K), 0.10 / (K-1))
        np.fill_diagonal(self.P, 0.90)

        #sample our moments
        base_mu = Y.mean(axis = 0)
        base_cov = np.cov(Y.T)
        std_devs = np.sqrt(np.diag(base_cov))

        self.intercepts = []
        self.coefs = []
        self.covs = []
        for k in range(K):
            #Regime 0: low var, maybe bull
            #Regime 1: higher var, maybe bear
            
            scale = 0.6 if k == 0 else 1.5
            shift = 0.25 * std_devs if k == 0 else -0.25 * std_devs
            self.intercepts.append(base_mu + shift)
            #small diagonal AR coefs
            self.coefs.append(np.eye(N) * 0.05)
            self.covs.append(base_cov * scale)

        old_llh = -np.inf

        for it in range(max_iter):
            # expectation step
            llh, filt, pred, dens = self._hamilton_filter(Y, X)
            smoothed, joint = self._kim_smoother(filt, pred)

            #check convergence
            if it > 0 and abs(llh - old_llh) < tol:
                break
            old_llh = llh

            # maximization step
            #transition matrix P, P_ij = expected transitions i -> j / expected visits to i
            sum_joint = joint[1:].sum(axis=0) #(K, K)

            for i in range(K):
                row = sum_joint[i]
                row_sum = row.sum()

                #catch case for no expected transitions
                if row_sum > 0:
                    self.P[i] = row/row_sum
                else:
                    self.P[i] = np.ones(K) / K

            #VAR params (moved outside the i loop to compute once per EM iteration)
            #hstacks stacks arrays horizontally
            Z = np.hstack([np.ones((T, 1)), X]) #(T, N+1)

            for k in range(K):
                #weight matrix = diag(ξ_{t|T}(k))
                W = np.diag(smoothed[:, k])

                # Weighted least squares: (Z' W Z) B = Z' W Y
                #try to assign weights to regressed probs & params for recursion
                lhs = Z.T @ W @ Z
                rhs = Z.T @ W @ Y
                #ridge penalty for numerical stability
                lhs.flat[::lhs.shape[0] + 1] += 1e-8
                try: 
                    B = np.linalg.solve(lhs, rhs) #(N+1, N)
                    self.intercepts[k] = B[0] #c_k
                    self.coefs[k] = B[1:].T #A_k
                except np.linalg.LinAlgError:
                    pass

                # Covariance matrix with Ledoit-Wolf shrinkage
                residuals = Y - (self.intercepts[k] + X @ self.coefs[k].T)
                w_sum = smoothed[:, k].sum() or 1e-15
                mle_cov = (residuals.T @ W @ residuals) / w_sum
                target_var = np.trace(mle_cov) / N
                identity_target = np.eye(N) * target_var
                #shrink toward identity matrix
                self.covs[k] = (
                    (1 - shrinkage) * mle_cov + shrinkage * identity_target
                )
        #store our EM probabilities
        self.filtered_probs = filt 
        self.smoothed_probs = smoothed

    def impulse_response(self, shock, horizon=20, xi0=None):
        """
        Overview: Kole van Dijk's findings were that we have a closed form
        solution for impluse response functions on MSIAH-VARs with regime transition
        states. This should theoretically allow us to better understand the impact
        of innovations/shocks on our chosen assets.
        Generalized impulse response using Kole van Dijk (2023) method
        shock: (N, ) vector: (e.g., one std dev. shock to VIX)
        horizon: number of steps ahead xi0: (K,) init regime probs (ergodic if none)
        returns: (horizon+1, N) array of expected responses
        """
        if xi0 is None:
            xi0 = self._ergodic_probs(self.P)
        N = self.N
        K = self.K
        responses = np.zeros((horizon+1, N))
        responses[0] = shock #impact(h=0)

        xi = xi0.copy()
        for h in range(1, horizon + 1):
            resp = np.zeros(N)
            #for each regime k, propogate shock linearly, weight by 
            #prob of being in regime k at horizon h
            for k in range(K):
                #A_k^h * shock
                A_pow = np.linalg.matrix_power(self.coefs[k], h)
                resp += xi[k] * (A_pow @ shock)
            responses[h] = resp
            #update regime dist: ξ_h = P' ξ_{h-1}
            xi = self.P.T @ xi
        return responses
    
    def unconditional_mean(self):
        """
        Overview: KvD relies on ergodicity of moments. This does mean we 
        have to assume i.i.d. in our assets. This could prove to be an issue
        no lie, but it is theoretically rational that the time average of our
        series follows the ensemble average. 
        Compute unconditional mean of the process
        Formula: μ = Σ_k ξ*_k (I - A_k)^{-1} c_k
        """
        erg = self._ergodic_probs(self.P)
        mu = np.zeros(self.N)
        for k in range(self.K):
            try:
                # (I - A_k)^{-1} c_k
                mu += erg[k] * np.linalg.solve(
                    np.eye(self.N) - self.coefs[k], 
                    self.intercepts[k]
                )
            except np.linalg.LinAlgError:
                #fallback if we can't invert
                mu += erg[k] * self.intercepts[k]
        return mu
    
    def simulate_irf(self, shock, horizon=20, xi0=None, n_sim=500):
        """
        Monte Carlo impulse response with 90% confidence bands.
        Returns mean, lower, upper arrays of shape (horizon+1, N).
        """
        if xi0 is None:
            xi0 = self._ergodic_probs(self.P)
        N = self.N
        K = self.K
        responses = np.zeros((n_sim, horizon+1, N))
        for s in range(n_sim):
            regime = np.random.choice(K, p=xi0)
            y_base = np.zeros(N)
            y_shock = shock.copy()
            responses[s, 0] = shock
            for h in range(1, horizon+1):
                regime = np.random.choice(K, p=self.P[regime])
                e = np.random.multivariate_normal(np.zeros(N), self.covs[regime])
                y_base = self.intercepts[regime] + self.coefs[regime] @ y_base + e
                y_shock = self.intercepts[regime] + self.coefs[regime] @ y_shock + e
                responses[s, h] = y_shock - y_base
        mean_irf = np.mean(responses, axis=0)
        lower_irf = np.percentile(responses, 5, axis=0)
        upper_irf = np.percentile(responses, 95, axis=0)
        return mean_irf, lower_irf, upper_irf
    

        
#integrated test
# (same imports and class as above, then:)

if __name__ == "__main__":
    np.random.seed(42)
    T = 1000
    # --- True parameters (ensuring stationarity) ---
    P_true = np.array([[0.95, 0.05], [0.2, 0.8]])

    # Regime 0: calm, all eigenvalues < 1
    c0 = np.array([0.01, 0.005, -0.002])
    A0 = np.array([[0.1, -0.02, 0.01],
                   [0.05, 0.05, -0.01],
                   [-0.1, 0.0, 0.5]])   # max eigenvalue ~0.5
    Sigma0 = np.diag([0.001, 0.0008, 0.002])   # diagonal for simplicity

    # Regime 1: turbulent, eigenvalues < 1
    c1 = np.array([-0.02, 0.01, 0.02])
    A1 = np.array([[0.0, 0.1, -0.3],
                   [-0.15, 0.2, 0.2],
                   [0.0, -0.1, 0.3]])   # max eigenvalue ~0.45
    Sigma1 = np.diag([0.005, 0.003, 0.006])

    # Simulate
    states = np.zeros(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = np.random.choice([0, 1], p=P_true[states[t-1]])

    Y = np.zeros((T, 3))
    Y[0] = c0 + np.random.multivariate_normal([0,0,0], Sigma0)
    for t in range(1, T):
        if states[t] == 0:
            Y[t] = c0 + A0 @ Y[t-1] + np.random.multivariate_normal([0,0,0], Sigma0)
        else:
            Y[t] = c1 + A1 @ Y[t-1] + np.random.multivariate_normal([0,0,0], Sigma1)

    # Fit
    model = MSIAHVAR(k_regimes=2)
    model.fit(Y, max_iter=50, tol=1e-4, shrinkage=0.05)

    print("Estimated transition matrix:")
    print(model.P.round(3))
    print("\nFiltered bear prob (last 10 days):")
    print(model.filtered_probs[-10:, 1].round(3))

    shock = np.array([0.0, 0.0, 0.1])
    irf = model.impulse_response(shock, horizon=10)
    print("\nImpulse response to VIX shock (SPY, TLT, VIX) – first 5 horizons:")
    print(irf[:5].round(4))

    # Check filtered probs sum to 1
    assert np.allclose(model.filtered_probs.sum(axis=1), 1.0, atol=1e-6)
    print("\nAll tests passed.")